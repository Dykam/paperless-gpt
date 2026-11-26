package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// Cooldown before a failed document is attempted again: the base delay after
// the first failure, doubling per consecutive failure up to the maximum.
var (
	documentRetryBaseDelay = 5 * time.Minute
	documentRetryMaxDelay  = time.Hour
)

// documentFailureTracker counts consecutive processing failures per document,
// so a poll loop can give up on a document that keeps failing, and gives each
// failed document a growing cooldown so the rest of the queue keeps moving.
// Used by the auto-OCR poll (ocrMaxRetries) and the auto-tag poll
// (autoTagMaxRetries). In-memory only: a restart grants a persistently failing
// document another round of attempts. The zero value is ready to use.
type documentFailureTracker struct {
	mu       sync.Mutex
	failures map[int]documentFailure
	now      func() time.Time // nil means time.Now
}

type documentFailure struct {
	count         int
	retryAt       time.Time
	serviceStreak int // consecutive service-unavailable attempts since the last counted failure
}

func documentRetryDelay(attempt int) time.Duration {
	delay := documentRetryBaseDelay
	for i := 1; i < attempt && delay < documentRetryMaxDelay; i++ {
		delay *= 2
	}
	if delay > documentRetryMaxDelay {
		delay = documentRetryMaxDelay
	}
	return delay
}

func (t *documentFailureTracker) clock() time.Time {
	if t.now != nil {
		return t.now()
	}
	return time.Now()
}

func (t *documentFailureTracker) entry(documentID int) documentFailure {
	if t.failures == nil {
		t.failures = make(map[int]documentFailure)
	}
	return t.failures[documentID]
}

func (t *documentFailureTracker) recordFailure(documentID int) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	f := t.entry(documentID)
	f.count++
	f.serviceStreak = 0
	f.retryAt = t.clock().Add(documentRetryDelay(f.count))
	t.failures[documentID] = f
	return f.count
}

// setAside holds a document back after a service-unavailable error without
// counting the attempt, with the same growing delay as a counted failure. A
// document that itself crashes the service is then retried at most hourly,
// but never fail-tagged.
func (t *documentFailureTracker) setAside(documentID int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f := t.entry(documentID)
	f.serviceStreak++
	if retryAt := t.clock().Add(documentRetryDelay(f.serviceStreak)); f.retryAt.Before(retryAt) {
		f.retryAt = retryAt
	}
	t.failures[documentID] = f
}

// coolingDown reports whether a document must not be attempted before the
// returned time.
func (t *documentFailureTracker) coolingDown(documentID int) (time.Time, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f, ok := t.failures[documentID]
	if !ok || !t.clock().Before(f.retryAt) {
		return time.Time{}, false
	}
	return f.retryAt, true
}

// reset clears the failure count for a document. Entries for documents whose
// trigger tag was removed externally before reaching the limit are never
// cleaned up; that leak is bounded by the number of such documents and
// accepted for simplicity.
func (t *documentFailureTracker) reset(documentID int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.failures, documentID)
}

// This is our interface, allowing us to enable proper testing
type BackgroundProcessor interface {
	processAutoOcrTagDocuments(ctx context.Context) (int, error)
	processAutoTagDocuments(ctx context.Context) (int, error)
	isOcrEnabled() bool
}

// Start our background tasks in a goroutine
func StartBackgroundTasks(ctx context.Context, app BackgroundProcessor) {
	go func() {
		backoff := pollBackoff{min: 10 * time.Second, max: 15 * time.Minute, interval: 10 * time.Second}

		for {
			select {
			case <-ctx.Done():
				log.Infoln("Background tasks shutting down")
				return
			default: // needed to make this non-blocking
			}

			processedCount, err := runPollRound(ctx, app)
			delay := backoff.next(processedCount, err)
			if err != nil {
				if onlyDocumentErrors(err) {
					log.Warnf("Some documents failed and will be retried after their cooldown: %v", err)
				} else {
					log.Errorf("Error in background tagging, next attempt in %v: %v", delay, err)
				}
			}

			if delay > 0 {
				select {
				case <-ctx.Done():
					log.Infoln("Background tasks shutting down")
					return
				case <-time.After(delay):
				}
			}
		}
	}()
}

// runPollRound runs the auto-OCR poll and then the auto-tag poll once. Both
// run even when the first reports document failures; only a failure that is
// not tied to one document (e.g. an unreachable service) skips the second.
func runPollRound(ctx context.Context, app BackgroundProcessor) (int, error) {
	count := 0
	var errs []error

	if app.isOcrEnabled() {
		ocrCount, err := app.processAutoOcrTagDocuments(ctx)
		count += ocrCount
		if err != nil {
			errs = append(errs, fmt.Errorf("error in processAutoOcrTagDocuments: %w", err))
			if !onlyDocumentErrors(err) {
				return count, errors.Join(errs...)
			}
		}
	}

	autoCount, err := app.processAutoTagDocuments(ctx)
	count += autoCount
	if err != nil {
		errs = append(errs, fmt.Errorf("error in processAutoTagDocuments: %w", err))
	}
	return count, errors.Join(errs...)
}

// serviceUnavailable wraps err so the poll loop recognises it as a reason to
// back off globally rather than a document failure.
func serviceUnavailable(err error) error {
	return fmt.Errorf("%w: %w", errServiceUnavailable, err)
}

// applyFailTagAfterPartialSuccess applies the fail tag to a document whose
// update succeeded only after paperless-gpt had to drop one or more fields
// rejected by paperless-ngx (see UpdateDocuments' strip-and-retry path).
//
// The document's tags in paperless-ngx have already been updated by the
// successful retry to whatever the LLM suggested (the auto tag is no longer
// present). To avoid clobbering those LLM-suggested tags, this function
// re-fetches the document's current state, then PATCHes only the tags field
// to append the fail tag.
//
// This is best-effort: if the re-fetch or the PATCH fails, the dropped-field
// information is logged but the document is left with no fail tag. The loop
// is still broken (the successful retry removed the auto tag) — only the
// user-visible marker is missing.
func applyFailTagAfterPartialSuccess(ctx context.Context, client ClientInterface, db *gorm.DB, documentID int, droppedFields []string) {
	docLogger := documentLogger(documentID)
	if failTag == "" {
		docLogger.Warnf("Document %d update succeeded after paperless-ngx rejected fields %v; no FAIL_TAG is configured, so the document is not marked for review.", documentID, droppedFields)
		return
	}
	currentDoc, err := client.GetDocument(ctx, documentID)
	if err != nil {
		docLogger.Errorf("Document %d update succeeded after dropping fields %v, but fetching current state to apply fail tag failed: %v", documentID, droppedFields, err)
		return
	}
	if slices.Contains(currentDoc.Tags, failTag) {
		docLogger.Warnf("Document %d update succeeded after dropping fields %v; fail tag %q is already present.", documentID, droppedFields, failTag)
		return
	}
	suggestion := DocumentSuggestion{
		ID:               documentID,
		OriginalDocument: currentDoc,
		SuggestedTags:    []string{failTag},
		KeepOriginalTags: true,
	}
	if err := client.UpdateDocuments(ctx, []DocumentSuggestion{suggestion}, db, false); err != nil {
		docLogger.Errorf("Document %d update succeeded after dropping fields %v, but applying fail tag %q failed: %v", documentID, droppedFields, failTag, err)
		return
	}
	docLogger.Warnf("Document %d update succeeded after paperless-ngx rejected fields %v; fail tag %q applied for user review.", documentID, droppedFields, failTag)
}

// markProcessingFailed performs a minimal tag-only PATCH that removes the
// trigger tag a document was picked up by (so the document is not
// re-processed on every poll cycle, which can cost real money on paid
// LLM/OCR providers) and, if failTag is configured, adds it as a marker so
// the user can find and review failed documents.
//
// The PATCH only manipulates tags and therefore should succeed even when the
// failure that led here was a field-validation rejection (e.g. an
// LLM-suggested date that is not a real calendar date). The reason string is
// used in log output only (e.g. "update failed", "OCR failed 3 times").
func markProcessingFailed(ctx context.Context, client ClientInterface, db *gorm.DB, document Document, removeTag string, reason string) error {
	docLogger := documentLogger(document.ID)
	recoveryFields := DocumentSuggestion{
		ID:               document.ID,
		OriginalDocument: document,
		RemoveTags:       []string{removeTag},
	}
	if failTag != "" {
		recoveryFields.SuggestedTags = []string{failTag}
		recoveryFields.KeepOriginalTags = true
	}
	if err := client.UpdateDocuments(ctx, []DocumentSuggestion{recoveryFields}, db, false); err != nil {
		return err
	}
	if failTag != "" {
		docLogger.Warnf("Document %d %s; %q tag removed and %q tag applied to break the processing loop.", document.ID, reason, removeTag, failTag)
	} else {
		docLogger.Warnf("Document %d %s; %q tag removed to break the processing loop (no failTag configured).", document.ID, reason, removeTag)
	}
	return nil
}

// recoverFromFailedUpdate is called when an UpdateDocuments call has failed for
// a document picked up by the auto-tagging or auto-OCR poll.
//
// On its own failure, this function logs at error level but does not return
// the error to the caller — the caller has already recorded the original
// update failure and the recovery is best-effort.
func recoverFromFailedUpdate(ctx context.Context, client ClientInterface, db *gorm.DB, document Document, removeTag string) {
	if err := markProcessingFailed(ctx, client, db, document, removeTag, "update failed"); err != nil {
		documentLogger(document.ID).Errorf("Recovery update for failed document %d also failed: %v. The %q tag may still be present and the document may be re-processed on the next poll cycle.", document.ID, err, removeTag)
	}
}

// processAutoTagDocuments handles the background auto-tagging of documents.
// It polls every tag the workflow router lists: the workflow trigger tags,
// then the global AUTO_TAG.
func (app *App) processAutoTagDocuments(ctx context.Context) (int, error) {
	totalProcessed := 0
	var allErrs []error
	for _, tag := range app.workflowRouter().PollTags() {
		n, err := app.processTagDocuments(ctx, tag)
		totalProcessed += n
		if err != nil {
			allErrs = append(allErrs, err)
			if errors.Is(err, errServiceUnavailable) {
				break
			}
		}
	}
	if len(allErrs) > 0 {
		return totalProcessed, errors.Join(allErrs...)
	}
	return totalProcessed, nil
}

// workflowOCRCoolingDown routes on the tag listing's document so a document
// waiting out a failed workflow OCR is skipped before it is fetched in full.
func (app *App) workflowOCRCoolingDown(doc Document, triggerTag string) (time.Time, bool) {
	wf := app.workflowRouter().WorkflowFor(doc, triggerTag)
	if wf == nil || !workflowWantsOCR(*wf) {
		return time.Time{}, false
	}
	return app.ocrFailures.coolingDown(doc.ID)
}

// processTagDocuments fetches documents with the given tag and runs the
// suggestion pipeline for each of them, with the workflow the router picks
// for the document or the global settings.
func (app *App) processTagDocuments(ctx context.Context, triggerTag string) (int, error) {
	documents, err := app.Client.GetDocumentsByTag(ctx, triggerTag, 25)
	if err != nil {
		err = fmt.Errorf("error fetching documents with tag %q: %w", triggerTag, err)
		if isServiceUnavailable(err) {
			return 0, serviceUnavailable(err)
		}
		return 0, err
	}

	if len(documents) == 0 {
		log.Debugf("No documents with tag %s found", triggerTag)
		return 0, nil // No documents to process
	}

	// Refresh the custom fields cache before processing, as we have documents
	refreshCustomFieldsCache(ctx, app.Client)

	log.Debugf("Found at least %d remaining documents with tag %s", len(documents), triggerTag)

	var errs []error
	processedCount := 0

	for _, docSummary := range documents {
		if retryAt, waiting := app.suggestionFailures.coolingDown(docSummary.ID); waiting {
			log.Debugf("Skipping document %d until %s after a failed suggestion attempt", docSummary.ID, retryAt.Format(time.RFC3339))
			continue
		}
		if retryAt, waiting := app.workflowOCRCoolingDown(docSummary, triggerTag); waiting {
			log.Debugf("Skipping document %d until %s after a failed workflow OCR attempt", docSummary.ID, retryAt.Format(time.RFC3339))
			continue
		}

		// Get the full document details, including custom fields
		document, err := app.Client.GetDocument(ctx, docSummary.ID)
		if err != nil {
			err = fmt.Errorf("error fetching full details for document %d: %w", docSummary.ID, err)
			documentLogger(docSummary.ID).Error(err.Error())
			if isServiceUnavailable(err) {
				app.suggestionFailures.setAside(docSummary.ID)
				errs = append(errs, serviceUnavailable(err))
				return processedCount, errors.Join(errs...)
			}
			errs = append(errs, documentError{err})
			continue
		}

		// Skip documents that have the autoOcrTag
		if slices.Contains(document.Tags, autoOcrTag) {
			log.Debugf("Skipping document %d as it has the OCR tag %s", document.ID, autoOcrTag)
			continue
		}

		docLogger := documentLogger(document.ID)
		workflow := app.workflowRouter().WorkflowFor(document, triggerTag)
		if workflow != nil {
			docLogger.Infof("Processing document for workflow %q (trigger tag: %s)", workflow.ID, triggerTag)
		} else {
			docLogger.Info("Processing document for auto-tagging")
		}

		// Optional per-workflow OCR before metadata generation.
		if workflow != nil {
			if wf := *workflow; workflowWantsOCR(wf) {
				if !app.isOcrEnabled() {
					// Retrying cannot help until the configuration changes, so
					// take the document out of the loop right away instead of
					// fetching it again on every poll.
					err = fmt.Errorf("workflow %q has enable_ocr but no OCR provider is configured", wf.ID)
					docLogger.Error(err.Error())
					if recErr := markProcessingFailed(ctx, app.Client, app.Database, document, triggerTag, "workflow OCR is enabled but no OCR provider is configured"); recErr != nil {
						errs = append(errs, documentError{fmt.Errorf("%w (removing the %q tag failed: %v)", err, triggerTag, recErr)})
					}
					continue
				}
				if retryAt, waiting := app.ocrFailures.coolingDown(document.ID); waiting {
					docLogger.Debugf("Skipping workflow OCR until %s after a failed attempt", retryAt.Format(time.RFC3339))
					continue
				}
				updatedDoc, ocrErr := app.runWorkflowDocumentOCR(ctx, document, wf, triggerTag, docLogger)
				if ocrErr != nil {
					if errors.Is(ocrErr, errServiceUnavailable) {
						errs = append(errs, ocrErr)
						return processedCount, errors.Join(errs...)
					}
					errs = append(errs, documentError{ocrErr})
					continue
				}
				if updatedDoc == nil {
					// OCR skipped or document left the queue (e.g. max retries).
					continue
				}
				document = *updatedDoc
			}
		}

		suggestionRequest := globalAutoProcessingRequest(document)
		suggestionRequest.IsAutoProcessing = true
		suggestionRequest.Workflow = workflow
		suggestionRequest.TriggerTag = triggerTag

		suggestions, err := app.generateDocumentSuggestions(ctx, suggestionRequest, docLogger)
		if err != nil {
			err = fmt.Errorf("error generating suggestions for document %d: %w", document.ID, err)
			docLogger.Error(err.Error())

			// Without a loop-break here the auto tag stays on the document and
			// it is picked up again every poll cycle, forever. A permanent LLM
			// error (a prompt that cannot fit the context, a model that no
			// longer exists) therefore re-bills the same request indefinitely,
			// and because GetDocumentsByTag fetches one unordered page, enough
			// such documents fill that page and starve every document behind
			// them — silently, since the container stays healthy. See #1071.
			if isServiceUnavailable(err) {
				app.suggestionFailures.setAside(document.ID)
				errs = append(errs, serviceUnavailable(err))
				return processedCount, errors.Join(errs...)
			}
			// A canceled poll-loop context is a shutdown, not a document
			// problem, so it must never count against the document. Same
			// reasoning as the auto-OCR poll below.
			if ctx.Err() == nil {
				attempts := app.suggestionFailures.recordFailure(document.ID)
				if autoTagMaxRetries > 0 && attempts >= autoTagMaxRetries {
					reason := fmt.Sprintf("suggestion generation failed %d times", attempts)
					if recErr := markProcessingFailed(ctx, app.Client, app.Database, document, triggerTag, reason); recErr != nil {
						// Keep the failure count so the tag removal is retried
						// once the cooldown ends.
						docLogger.Errorf("Removing %q tag after repeated suggestion failures also failed: %v. The document will be retried after its cooldown.", triggerTag, recErr)
						errs = append(errs, documentError{err})
						continue
					}
					app.suggestionFailures.reset(document.ID)
					continue
				}
			}
			errs = append(errs, documentError{err})
			continue
		}

		err = app.Client.UpdateDocuments(ctx, suggestions, app.Database, false)
		if err != nil {
			var partial *PartialUpdateError
			if errors.As(err, &partial) {
				// Update went through but paperless-ngx rejected some fields,
				// which UpdateDocuments dropped in order to land the rest.
				// The auto tag is already gone (it was part of the successful
				// retry's tag update). Apply the fail tag so the user sees
				// that this document needs review.
				applyFailTagAfterPartialSuccess(ctx, app.Client, app.Database, partial.DocumentID, partial.DroppedFields)
				app.suggestionFailures.reset(document.ID)
				processedCount++
				continue
			}
			err = fmt.Errorf("error updating document %d: %w", document.ID, err)
			docLogger.Error(err.Error())
			// Keep the trigger tag on an outage, so the document is processed
			// again once paperless-ngx is back.
			if isServiceUnavailable(err) {
				app.suggestionFailures.setAside(document.ID)
				errs = append(errs, serviceUnavailable(err))
				return processedCount, errors.Join(errs...)
			}
			// The cooldown matters when the recovery PATCH fails too and the
			// trigger tag stays.
			app.suggestionFailures.recordFailure(document.ID)
			errs = append(errs, documentError{err})
			recoverFromFailedUpdate(ctx, app.Client, app.Database, document, triggerTag)
			continue
		}

		app.suggestionFailures.reset(document.ID)
		docLogger.Info("Successfully processed document")
		processedCount++
	}

	if len(errs) > 0 {
		return processedCount, errors.Join(errs...)
	}

	return processedCount, nil
}

// runWorkflowDocumentOCR runs OCR for a workflow-tagged document, writes the
// extracted text back, and leaves the workflow trigger tag in place so the
// caller can continue with metadata generation. Returns nil,nil when the
// document should be skipped for this cycle (already complete / max retries).
func (app *App) runWorkflowDocumentOCR(ctx context.Context, document Document, wf WorkflowConfig, triggerTag string, docLogger *logrus.Entry) (*Document, error) {
	if app.pdfOCRTagging {
		for _, tag := range document.Tags {
			if tag == app.pdfOCRCompleteTag {
				docLogger.Infof("Document already has OCR complete tag '%s', skipping workflow OCR", app.pdfOCRCompleteTag)
				app.ocrFailures.reset(document.ID)
				return &document, nil
			}
		}
	}

	options := app.effectiveOCROptionsForWorkflow(wf)
	options.ExistingContent = document.Content

	jobID := generateJobID()
	jobStore.addJob(&Job{
		ID:         jobID,
		DocumentID: document.ID,
		Status:     "in_progress",
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Options:    options,
	})
	trigger := "workflow:" + wf.ID
	if err := CreateOCRRun(app.Database, &OCRRun{
		JobID:           jobID,
		DocumentID:      document.ID,
		DocumentTitle:   document.Title,
		Trigger:         trigger,
		LimitPages:      options.LimitPages,
		ProcessMode:     options.ProcessMode,
		UploadPDF:       options.UploadPDF,
		ReplaceOriginal: options.ReplaceOriginal,
		CopyMetadata:    options.CopyMetadata,
		PromptOverride:  options.PromptOverride,
		Provider:        app.ocrProviderLabel,
		PDFAction:       "none",
	}); err != nil {
		docLogger.Warnf("Failed to persist workflow OCR run: %v", err)
	}

	var processedDoc *ProcessedDocument
	var err error
	if app.docProcessor != nil {
		processedDoc, err = app.docProcessor.ProcessDocumentOCR(ctx, document.ID, options, jobID)
	} else {
		processedDoc, err = app.ProcessDocumentOCR(ctx, document.ID, options, jobID)
	}

	pagesDone, totalPages := jobStore.progress(jobID)
	if err != nil {
		docLogger.Errorf("Workflow OCR processing failed: %v", err)
		jobStore.updateJobStatus(jobID, "failed", err.Error())
		finishOCRRunLogged(app, jobID, "failed", err.Error(), pagesDone, totalPages, "", "")

		if isServiceUnavailable(err) {
			app.ocrFailures.setAside(document.ID)
			return nil, serviceUnavailable(fmt.Errorf("document %d workflow OCR error: %w", document.ID, err))
		}
		if ctx.Err() == nil {
			attempts := app.ocrFailures.recordFailure(document.ID)
			if ocrMaxRetries > 0 && attempts >= ocrMaxRetries {
				reason := fmt.Sprintf("OCR failed %d times", attempts)
				if recErr := markProcessingFailed(ctx, app.Client, app.Database, document, triggerTag, reason); recErr != nil {
					docLogger.Errorf("Removing %q tag after repeated OCR failures also failed: %v", triggerTag, recErr)
					return nil, fmt.Errorf("document %d workflow OCR error: %w", document.ID, err)
				}
				app.ocrFailures.reset(document.ID)
				return nil, nil
			}
		}
		return nil, fmt.Errorf("document %d workflow OCR error: %w", document.ID, err)
	}

	if processedDoc == nil {
		app.ocrFailures.reset(document.ID)
		docLogger.Info("Workflow OCR processing skipped for document")
		jobStore.updateJobStatus(jobID, "completed", "Skipped (already processed)")
		finishOCRRunLogged(app, jobID, "completed", "", pagesDone, totalPages, "none", "Skipped (already processed)")
		return &document, nil
	}
	jobStore.updateJobStatus(jobID, "completed", processedDoc.Text)
	finishOCRRunLogged(app, jobID, "completed", "", pagesDone, totalPages, processedDoc.PDFAction, processedDoc.PDFDetail)
	if err := PruneOCRRuns(app.Database, document.ID); err != nil {
		docLogger.Warnf("Failed to prune OCR runs: %v", err)
	}

	suggestion := DocumentSuggestion{
		ID:               document.ID,
		OriginalDocument: document,
		SuggestedContent: processedDoc.Text,
		AddTags: func() []string {
			if app.pdfOCRTagging {
				return []string{app.pdfOCRCompleteTag}
			}
			return nil
		}(),
	}
	if app.pdfOCRTagging && app.pdfOCRCompleteTag != "" {
		suggestion.SuggestedTags = []string{app.pdfOCRCompleteTag}
		suggestion.KeepOriginalTags = true
	}

	if err := app.Client.UpdateDocuments(ctx, []DocumentSuggestion{suggestion}, app.Database, false); err != nil {
		var partial *PartialUpdateError
		if errors.As(err, &partial) {
			applyFailTagAfterPartialSuccess(ctx, app.Client, app.Database, partial.DocumentID, partial.DroppedFields)
			app.ocrFailures.reset(document.ID)
			document.Content = processedDoc.Text
			return &document, nil
		}
		docLogger.Errorf("Update after workflow OCR failed: %v", err)
		err = fmt.Errorf("document %d update after workflow OCR error: %w", document.ID, err)
		if isServiceUnavailable(err) {
			app.ocrFailures.setAside(document.ID)
			return nil, serviceUnavailable(err)
		}
		app.ocrFailures.recordFailure(document.ID)
		recoverFromFailedUpdate(ctx, app.Client, app.Database, document, triggerTag)
		return nil, err
	}

	app.ocrFailures.reset(document.ID)
	document.Content = processedDoc.Text
	if app.pdfOCRTagging && app.pdfOCRCompleteTag != "" && !slices.Contains(document.Tags, app.pdfOCRCompleteTag) {
		document.Tags = append(document.Tags, app.pdfOCRCompleteTag)
	}
	docLogger.Info("Successfully processed workflow OCR")
	return &document, nil
}

// processAutoOcrTagDocuments handles the background auto-tagging of OCR documents
func (app *App) processAutoOcrTagDocuments(ctx context.Context) (int, error) {
	documents, err := app.Client.GetDocumentsByTag(ctx, autoOcrTag, 25)
	if err != nil {
		err = fmt.Errorf("error fetching documents with autoOcrTag: %w", err)
		if isServiceUnavailable(err) {
			return 0, serviceUnavailable(err)
		}
		return 0, err
	}

	if len(documents) == 0 {
		log.Debugf("No documents with tag %s found", autoOcrTag)
		return 0, nil
	}

	log.Debugf("Found %d documents with tag %s", len(documents), autoOcrTag)

	successCount := 0
	var errs []error

	for _, document := range documents {
		docLogger := documentLogger(document.ID)
		if retryAt, waiting := app.ocrFailures.coolingDown(document.ID); waiting {
			docLogger.Debugf("Skipping OCR until %s after a failed attempt", retryAt.Format(time.RFC3339))
			continue
		}
		docLogger.Info("Processing document for OCR")

		// Skip OCR if the document already has the OCR complete tag and tagging is enabled
		if app.pdfOCRTagging {
			hasCompleteTag := false
			for _, tag := range document.Tags {
				if tag == app.pdfOCRCompleteTag {
					hasCompleteTag = true
					break
				}
			}

			if hasCompleteTag {
				docLogger.Infof("Document already has OCR complete tag '%s', skipping OCR processing", app.pdfOCRCompleteTag)

				// Remove only the autoOcrTag to take it out of the processing queue
				// while preserving the OCR complete tag
				err = app.Client.UpdateDocuments(ctx, []DocumentSuggestion{
					{
						ID:               document.ID,
						OriginalDocument: document,
						RemoveTags:       []string{autoOcrTag},
					},
				}, app.Database, false)

				if err != nil {
					docLogger.Errorf("Update to remove autoOcrTag failed: %v", err)
					err = fmt.Errorf("document %d update error: %w", document.ID, err)
					if isServiceUnavailable(err) {
						app.ocrFailures.setAside(document.ID)
						errs = append(errs, serviceUnavailable(err))
						return successCount, fmt.Errorf("one or more errors occurred: %w", errors.Join(errs...))
					}
					app.ocrFailures.recordFailure(document.ID)
					errs = append(errs, documentError{err})
					continue
				}

				docLogger.Info("Successfully removed auto OCR tag")
				successCount++
				continue
			}
		}

		options := app.effectiveOCRDefaults()
		options.ExistingContent = document.Content

		// Every auto run is tracked exactly like a manual one: registered in
		// the in-memory job store (live progress) and persisted as an OCR Run
		// (Activity log), so hands-off users can audit what happened.
		jobID := generateJobID()
		jobStore.addJob(&Job{
			ID:         jobID,
			DocumentID: document.ID,
			Status:     "in_progress",
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			Options:    options,
		})
		if err := CreateOCRRun(app.Database, &OCRRun{
			JobID:           jobID,
			DocumentID:      document.ID,
			DocumentTitle:   document.Title,
			Trigger:         "auto",
			LimitPages:      options.LimitPages,
			ProcessMode:     options.ProcessMode,
			UploadPDF:       options.UploadPDF,
			UploadMode:      options.UploadMode,
			ReplaceOriginal: options.ReplaceOriginal,
			CopyMetadata:    options.CopyMetadata,
			Provider:        app.ocrProviderLabel,
			PDFAction:       "none",
		}); err != nil {
			docLogger.Warnf("Failed to persist auto OCR run: %v", err)
		}

		// Use the DocumentProcessor interface instead of calling the method directly
		var processedDoc *ProcessedDocument
		var err error
		if app.docProcessor != nil {
			// Use injected processor if available
			processedDoc, err = app.docProcessor.ProcessDocumentOCR(ctx, document.ID, options, jobID)
		} else {
			// Use the app's own implementation if no processor is injected
			processedDoc, err = app.ProcessDocumentOCR(ctx, document.ID, options, jobID)
		}

		pagesDone, totalPages := jobStore.progress(jobID)
		if err != nil {
			docLogger.Errorf("OCR processing failed: %v", err)
			jobStore.updateJobStatus(jobID, "failed", err.Error())
			finishOCRRunLogged(app, jobID, "failed", err.Error(), pagesDone, totalPages, "", "")

			err = fmt.Errorf("document %d OCR error: %w", document.ID, err)
			if isServiceUnavailable(err) {
				app.ocrFailures.setAside(document.ID)
				errs = append(errs, serviceUnavailable(err))
				return successCount, fmt.Errorf("one or more errors occurred: %w", errors.Join(errs...))
			}
			// A canceled/expired poll-loop context is a shutdown, not a
			// document problem — never count it against the document.
			// Deliberately checked on ctx rather than via errors.Is on err:
			// providers wrap their calls in their own timeout contexts
			// (e.g. Azure, ocr/azure_provider.go), and those per-document
			// timeouts must count toward the limit.
			if ctx.Err() == nil {
				attempts := app.ocrFailures.recordFailure(document.ID)
				if ocrMaxRetries > 0 && attempts >= ocrMaxRetries {
					reason := fmt.Sprintf("OCR failed %d times", attempts)
					if recErr := markProcessingFailed(ctx, app.Client, app.Database, document, autoOcrTag, reason); recErr != nil {
						// Keep the failure count so the tag removal is
						// retried once the cooldown ends.
						docLogger.Errorf("Removing %q tag after repeated OCR failures also failed: %v. The document will be retried after its cooldown.", autoOcrTag, recErr)
						errs = append(errs, documentError{err})
						continue
					}
					app.ocrFailures.reset(document.ID)
					continue
				}
			}
			errs = append(errs, documentError{err})
			continue
		}
		if processedDoc == nil {
			app.ocrFailures.reset(document.ID)
			docLogger.Info("OCR processing skipped for document")
			jobStore.updateJobStatus(jobID, "completed", "Skipped (already processed)")
			finishOCRRunLogged(app, jobID, "completed", "", pagesDone, totalPages, "none", "Skipped (already processed)")
			continue
		}
		jobStore.updateJobStatus(jobID, "completed", processedDoc.Text)
		finishOCRRunLogged(app, jobID, "completed", "", pagesDone, totalPages, processedDoc.PDFAction, processedDoc.PDFDetail)
		if err := PruneOCRRuns(app.Database, document.ID); err != nil {
			docLogger.Warnf("Failed to prune OCR runs: %v", err)
		}
		docLogger.Debug("OCR processing completed")

		documentSuggestion := DocumentSuggestion{
			ID:               document.ID,
			OriginalDocument: document,
			SuggestedContent: processedDoc.Text,
			RemoveTags:       []string{autoOcrTag},
			// Add OCR complete tag if tagging is enabled and PDF wasn't uploaded (upload handles tagging)
			AddTags: func() []string {
				if app.pdfOCRTagging && !options.UploadPDF {
					return []string{app.pdfOCRCompleteTag}
				}
				return nil
			}(),
		}

		if (app.pdfOCRTagging) && app.pdfOCRCompleteTag != "" {
			// Add the OCR complete tag if tagging is enabled
			documentSuggestion.SuggestedTags = []string{app.pdfOCRCompleteTag}
			documentSuggestion.KeepOriginalTags = true
			docLogger.Infof("Adding OCR complete tag '%s'", app.pdfOCRCompleteTag)
		}

		// Skip updating the original document if it was actually replaced (deleted) during OCR.
		// The replacement document will be processed as a new document on the next cycle.
		if options.ReplaceOriginal && processedDoc != nil && processedDoc.ReplacedOriginal {
			docLogger.Info("Skipping tag update for replaced document (original was deleted)")
		} else {
			err = app.Client.UpdateDocuments(ctx, []DocumentSuggestion{
				documentSuggestion,
			}, app.Database, false)
			if err != nil {
				var partial *PartialUpdateError
				if errors.As(err, &partial) {
					applyFailTagAfterPartialSuccess(ctx, app.Client, app.Database, partial.DocumentID, partial.DroppedFields)
					app.ocrFailures.reset(document.ID)
					// Treat as a (partial) success: tag was removed, fail tag applied.
					docLogger.Info("Successfully processed document OCR (with partial-update fail-tag marker)")
					successCount++
					continue
				}
				docLogger.Errorf("Update after OCR failed: %v", err)
				err = fmt.Errorf("document %d update error: %w", document.ID, err)
				if isServiceUnavailable(err) {
					app.ocrFailures.setAside(document.ID)
					errs = append(errs, serviceUnavailable(err))
					return successCount, fmt.Errorf("one or more errors occurred: %w", errors.Join(errs...))
				}
				// The cooldown matters when the recovery PATCH fails too and the
				// trigger tag stays.
				app.ocrFailures.recordFailure(document.ID)
				errs = append(errs, documentError{err})
				recoverFromFailedUpdate(ctx, app.Client, app.Database, document, autoOcrTag)
				continue
			}
		}

		app.ocrFailures.reset(document.ID)
		docLogger.Info("Successfully processed document OCR")
		successCount++
	}

	if len(errs) > 0 {
		return successCount, fmt.Errorf("one or more errors occurred: %w", errors.Join(errs...))
	}

	return successCount, nil
}
