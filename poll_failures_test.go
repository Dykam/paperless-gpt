package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// disableDocumentCooldown lets a test retry a failed document in the very next
// round, for tests about retry counting rather than retry timing.
func disableDocumentCooldown(t *testing.T) {
	t.Helper()
	prev := documentRetryBaseDelay
	documentRetryBaseDelay = 0
	t.Cleanup(func() { documentRetryBaseDelay = prev })
}

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func TestIsServiceUnavailable(t *testing.T) {
	dialRefused := &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"wrapped dial error", fmt.Errorf("HTTP request failed: %w", &url.Error{Op: "Get", URL: "http://paperless/api/", Err: dialRefused}), true},
		{"DNS failure", &url.Error{Op: "Get", URL: "http://paperless/api/", Err: &net.DNSError{Err: "no such host", Name: "paperless"}}, true},
		{"connection reset", fmt.Errorf("read: %w", syscall.ECONNRESET), true},
		{"refused, as text only", errors.New(`Post "http://ollama:11434/api/chat": dial tcp 10.0.0.1:11434: connect: connection refused`), true},
		{"LLM 503", errors.New("error getting response from LLM: API returned unexpected status code: 503: service unavailable"), true},
		{"LLM 429", errors.New("all retry attempts failed, last error: API returned unexpected status code: 429: rate limited"), true},
		{"paperless 502", errors.New("error fetching document 5: 502, <html>Bad Gateway</html>"), true},
		{"paperless search 504", errors.New("error searching documents: status=504, body="), true},
		{"Azure 503", errors.New("unexpected status code 503: busy"), true},
		{"Docling 502", errors.New("docling API returned status 502: bad gateway"), true},
		{"Mistral 429", errors.New("status: 429, response: rate limited"), true},
		{"Ollama queue full", errors.New("error getting response from LLM: server busy, please try again.  maximum pending requests exceeded"), true},
		{"gRPC unavailable", errors.New("rpc error: code = Unavailable desc = connection closed"), true},
		{"server closed the connection", errors.New(`error getting response from LLM: Post "http://ollama:11434/api/chat": EOF`), true},
		{"EOF inside url.Error", &url.Error{Op: "Post", URL: "http://ollama:11434/api/chat", Err: io.EOF}, true},
		{"broken pipe", fmt.Errorf("write: %w", syscall.EPIPE), true},
		{"truncated image", errors.New("document 7: decoding original image: unexpected EOF"), false},
		{"truncated image, wrapped sentinel", fmt.Errorf("document 7: decoding original image: %w", io.ErrUnexpectedEOF), false},
		{"model looped", errors.New("error performing OCR for document 253, page 3: error getting response from LLM: prediction aborted, token repeat limit reached"), false},
		{"LLM 500", errors.New("error getting response from LLM: API returned unexpected status code: 500: llama runner process has terminated"), false},
		{"paperless 404", errors.New("error fetching document 5: 404, not found"), false},
		{"document ID that looks like a status", errors.New("error downloading document 503: 404, not found"), false},
		{"provider timeout", fmt.Errorf("operation timed out after 2m0s: %w", context.DeadlineExceeded), false},
		{"shutdown", context.Canceled, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isServiceUnavailable(tc.err))
		})
	}
}

func TestOnlyDocumentErrors(t *testing.T) {
	docErr := documentError{errors.New("pdf is broken")}
	assert.False(t, onlyDocumentErrors(nil))
	assert.True(t, onlyDocumentErrors(docErr))
	assert.True(t, onlyDocumentErrors(errors.Join(docErr, documentError{errors.New("bad date")})))
	assert.True(t, onlyDocumentErrors(fmt.Errorf("one or more errors occurred: %w", errors.Join(docErr))))
	assert.False(t, onlyDocumentErrors(errors.Join(docErr, errors.New("listing failed"))))
	assert.False(t, onlyDocumentErrors(errors.Join(docErr, serviceUnavailable(errors.New("connection refused")))))
}

func TestPollBackoff(t *testing.T) {
	b := pollBackoff{min: 10 * time.Second, max: 40 * time.Second, interval: 3 * time.Second}
	docErr := documentError{errors.New("pdf is broken")}
	down := serviceUnavailable(errors.New("connection refused"))

	assert.Equal(t, 3*time.Second, b.next(0, nil), "idle round waits the poll interval")
	assert.Equal(t, time.Duration(0), b.next(2, nil), "productive round polls again at once")
	assert.Equal(t, 3*time.Second, b.next(0, docErr), "document failures never back off")
	assert.Equal(t, time.Duration(0), b.next(1, docErr))

	assert.Equal(t, 10*time.Second, b.next(0, down))
	assert.Equal(t, 20*time.Second, b.next(0, down))
	assert.Equal(t, 40*time.Second, b.next(0, down))
	assert.Equal(t, 40*time.Second, b.next(0, down), "capped at max")
	assert.Equal(t, 40*time.Second, b.next(3, down), "progress does not lift a service backoff by itself")

	assert.Equal(t, 3*time.Second, b.next(0, docErr), "a round without service errors ends the backoff")
	assert.Equal(t, 10*time.Second, b.next(0, down), "and the next outage starts from min again")
}

func TestDocumentFailureTracker_SetAside(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	tracker := documentFailureTracker{now: clock.now}

	tracker.setAside(1)
	retryAt, waiting := tracker.coolingDown(1)
	assert.True(t, waiting, "the document is set aside so the rest of the queue can be tried")
	assert.Equal(t, clock.t.Add(documentRetryBaseDelay), retryAt)

	clock.advance(documentRetryBaseDelay)
	tracker.setAside(1)
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(2*documentRetryBaseDelay), retryAt, "consecutive set-asides grow like failures")

	for i := 0; i < 10; i++ {
		tracker.setAside(1)
	}
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(documentRetryMaxDelay), retryAt, "capped, so a crashing document is retried at most hourly")

	assert.Equal(t, 1, tracker.recordFailure(1), "set-asides are never counted")
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(documentRetryBaseDelay), retryAt, "a counted failure uses its own delay")

	tracker.recordFailure(1)
	tracker.recordFailure(1)
	tracker.setAside(1)
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(4*documentRetryBaseDelay), retryAt, "a set-aside never shortens a longer cooldown")
}

func TestDocumentFailureTracker_Cooldown(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	tracker := documentFailureTracker{now: clock.now}

	_, waiting := tracker.coolingDown(1)
	assert.False(t, waiting, "unknown document is not cooling down")

	require.Equal(t, 1, tracker.recordFailure(1))
	retryAt, waiting := tracker.coolingDown(1)
	assert.True(t, waiting)
	assert.Equal(t, clock.t.Add(documentRetryBaseDelay), retryAt)

	clock.advance(documentRetryBaseDelay)
	_, waiting = tracker.coolingDown(1)
	assert.False(t, waiting, "cooldown ends at retryAt")

	require.Equal(t, 2, tracker.recordFailure(1))
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(2*documentRetryBaseDelay), retryAt, "cooldown doubles per consecutive failure")

	for i := 0; i < 10; i++ {
		tracker.recordFailure(1)
	}
	retryAt, _ = tracker.coolingDown(1)
	assert.Equal(t, clock.t.Add(documentRetryMaxDelay), retryAt, "cooldown is capped")

	_, waiting = tracker.coolingDown(2)
	assert.False(t, waiting, "documents cool down independently")

	tracker.reset(1)
	_, waiting = tracker.coolingDown(1)
	assert.False(t, waiting)
	assert.Equal(t, 1, tracker.recordFailure(1), "reset restarts the count")
}

func ocrTestApp(t *testing.T, client *recordingClient, processor *mockDocumentProcessor, clock *fakeClock) *App {
	app := &App{Client: client, docProcessor: processor, Database: newOCRRunTestDB(t)}
	app.ocrFailures.now = clock.now
	return app
}

func setOCRGlobals(t *testing.T, maxRetries int) {
	prevFailTag, prevAutoOcrTag, prevMaxRetries := failTag, autoOcrTag, ocrMaxRetries
	t.Cleanup(func() { failTag, autoOcrTag, ocrMaxRetries = prevFailTag, prevAutoOcrTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoOcrTag = "paperless-gpt-ocr-auto"
	ocrMaxRetries = maxRetries
}

func TestProcessAutoOcrTagDocuments_FailedDocumentWaitsForCooldown(t *testing.T) {
	setOCRGlobals(t, 3)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	broken := Document{ID: 1, Tags: []string{autoOcrTag}}
	healthy := Document{ID: 2, Tags: []string{autoOcrTag}}
	client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: {broken, healthy}}}
	processor := &mockDocumentProcessor{mockText: "text", perDocErr: map[int]error{1: errors.New("prediction aborted, token repeat limit reached")}}
	app := ocrTestApp(t, client, processor, clock)

	count, err := app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 1, count, "the healthy document is processed despite the broken one")
	require.Error(t, err)
	assert.True(t, onlyDocumentErrors(err), "a document failure must not trigger the global backoff")

	_, _ = app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 1, processor.ocrCalls[1], "the broken document is skipped while cooling down")

	clock.advance(documentRetryBaseDelay)
	_, _ = app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 2, processor.ocrCalls[1], "and retried once the cooldown ends")
}

func TestProcessAutoOcrTagDocuments_RetriesForeverStillCoolsDown(t *testing.T) {
	setOCRGlobals(t, 0)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: {{ID: 1, Tags: []string{autoOcrTag}}}}}
	processor := &mockDocumentProcessor{mockErr: errors.New("pdf is broken")}
	app := ocrTestApp(t, client, processor, clock)

	_, _ = app.processAutoOcrTagDocuments(context.Background())
	_, _ = app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 1, processor.ocrCalls[1], "OCR_MAX_RETRIES=0 removes the limit, not the cooldown")
	assert.Empty(t, client.calls)
}

func TestProcessAutoOcrTagDocuments_ServiceUnavailableStopsRoundUncounted(t *testing.T) {
	setOCRGlobals(t, 1)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	first := Document{ID: 1, Tags: []string{autoOcrTag}}
	second := Document{ID: 2, Tags: []string{autoOcrTag}}
	client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: {first, second}}}
	processor := &mockDocumentProcessor{mockText: "text", perDocErr: map[int]error{
		1: errors.New(`error getting response from LLM: Post "http://ollama:11434/api/chat": dial tcp 10.0.0.1:11434: connect: connection refused`),
	}}
	app := ocrTestApp(t, client, processor, clock)

	count, err := app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 0, count)
	require.Error(t, err)
	assert.ErrorIs(t, err, errServiceUnavailable)
	assert.False(t, onlyDocumentErrors(err), "an unreachable service must trigger the global backoff")
	assert.Zero(t, processor.ocrCalls[2], "the round stops instead of failing every remaining document")
	assert.Empty(t, client.calls, "the document is not fail-tagged even with OCR_MAX_RETRIES=1")

	clock.advance(documentRetryBaseDelay)
	_, err = app.processAutoOcrTagDocuments(context.Background())
	assert.ErrorIs(t, err, errServiceUnavailable, "with nothing succeeding in between it is still an outage")
	assert.Empty(t, client.calls, "an outage never fail-tags, however long it lasts")
}

func TestProcessAutoOcrTagDocuments_DocumentThatLooksLikeAnOutageIsSetAside(t *testing.T) {
	setOCRGlobals(t, 1)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	crasher := Document{ID: 1, Tags: []string{autoOcrTag}}
	healthy := Document{ID: 2, Tags: []string{autoOcrTag}}
	client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: {crasher, healthy}}}
	processor := &mockDocumentProcessor{mockText: "text", perDocErr: map[int]error{
		1: errors.New(`error getting response from LLM: Post "http://ollama:11434/api/chat": EOF`),
	}}
	app := ocrTestApp(t, client, processor, clock)

	_, err := app.processAutoOcrTagDocuments(context.Background())
	assert.ErrorIs(t, err, errServiceUnavailable)
	assert.Zero(t, processor.ocrCalls[2])

	clock.advance(time.Second)
	count, err := app.processAutoOcrTagDocuments(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, 1, count, "the set-aside document no longer blocks the one behind it")

	clock.advance(documentRetryBaseDelay)
	_, err = app.processAutoOcrTagDocuments(context.Background())
	assert.ErrorIs(t, err, errServiceUnavailable)
	assert.Equal(t, 2, processor.ocrCalls[1])
	assert.Empty(t, client.callsFor(1), "never fail-tagged, even with OCR_MAX_RETRIES=1")
	retryAt, _ := app.ocrFailures.coolingDown(1)
	assert.Equal(t, clock.t.Add(2*documentRetryBaseDelay), retryAt, "each repeat waits longer")
}

func TestProcessAutoOcrTagDocuments_FailedUpdateAndRecoveryCoolsDown(t *testing.T) {
	setOCRGlobals(t, 3)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoOcrTag: {{ID: 1, Tags: []string{autoOcrTag}}}},
		updateErr:       errors.New("error updating document 1: 403, forbidden"),
		failAfterNCalls: 100,
	}
	processor := &mockDocumentProcessor{mockText: "text"}
	app := ocrTestApp(t, client, processor, clock)

	_, err := app.processAutoOcrTagDocuments(context.Background())
	assert.True(t, onlyDocumentErrors(err))
	require.Len(t, client.calls, 2, "the update and the recovery PATCH, both rejected")

	_, _ = app.processAutoOcrTagDocuments(context.Background())
	assert.Equal(t, 1, processor.ocrCalls[1], "the trigger tag is still there, but OCR is not paid for again right away")
}

func TestProcessAutoOcrTagDocuments_UnreachablePaperlessKeepsTriggerTag(t *testing.T) {
	setOCRGlobals(t, 3)
	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoOcrTag: {{ID: 1, Tags: []string{autoOcrTag}}}},
		updateErr:       fmt.Errorf("HTTP request failed: %w", &url.Error{Op: "Patch", URL: "http://paperless/api/documents/1/", Err: &net.OpError{Op: "dial", Net: "tcp", Err: syscall.ECONNREFUSED}}),
		failAfterNCalls: 1,
	}
	app := ocrTestApp(t, client, &mockDocumentProcessor{mockText: "text"}, clock)

	_, err := app.processAutoOcrTagDocuments(context.Background())
	assert.ErrorIs(t, err, errServiceUnavailable)
	require.Len(t, client.calls, 1, "only the update itself; no recovery PATCH that would fail-tag the document")
}

func TestProcessAutoTagDocuments_ServiceUnavailableNotCounted(t *testing.T) {
	prevFailTag, prevAutoTag, prevMaxRetries := failTag, autoTag, autoTagMaxRetries
	t.Cleanup(func() { failTag, autoTag, autoTagMaxRetries = prevFailTag, prevAutoTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	autoTagMaxRetries = 1
	isolateWorkflowSettings(t)

	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoTag: {{ID: 5, Tags: []string{autoTag}}}},
		getAllTagsErr:   errors.New(`Get "http://paperless/api/tags/": dial tcp: lookup paperless: no such host`),
	}
	app := &App{Client: client}
	app.suggestionFailures.now = clock.now

	for round := 1; round <= 3; round++ {
		_, err := app.processAutoTagDocuments(context.Background())
		assert.ErrorIs(t, err, errServiceUnavailable, "round %d", round)
		clock.advance(documentRetryMaxDelay)
	}
	assert.Empty(t, client.calls, "connection failures never use up the document's attempts")
}

func TestProcessAutoTagDocuments_FailedDocumentWaitsForCooldown(t *testing.T) {
	prevFailTag, prevAutoTag, prevMaxRetries := failTag, autoTag, autoTagMaxRetries
	t.Cleanup(func() { failTag, autoTag, autoTagMaxRetries = prevFailTag, prevAutoTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	autoTagMaxRetries = 2
	isolateWorkflowSettings(t)

	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoTag: {{ID: 6, Tags: []string{autoTag}}}},
		getAllTagsErr:   errors.New("exceeds the available context size"),
	}
	app := &App{Client: client}
	app.suggestionFailures.now = clock.now

	_, err := app.processAutoTagDocuments(context.Background())
	assert.True(t, onlyDocumentErrors(err))

	count, err := app.processAutoTagDocuments(context.Background())
	assert.NoError(t, err, "a document in cooldown is skipped, not failed again")
	assert.Equal(t, 0, count)
	assert.Empty(t, client.calls)

	clock.advance(documentRetryBaseDelay)
	_, _ = app.processAutoTagDocuments(context.Background())
	require.Len(t, client.calls, 1, "second attempt after the cooldown reaches the limit")
	assert.Equal(t, []string{failTag}, client.calls[0].SuggestedTags)
}

func TestProcessTagDocuments_WorkflowOCRFailureKinds(t *testing.T) {
	prevFailTag, prevAutoTag, prevMaxRetries := failTag, autoTag, ocrMaxRetries
	t.Cleanup(func() { failTag, autoTag, ocrMaxRetries = prevFailTag, prevAutoTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	ocrMaxRetries = 3
	isolateWorkflowSettings(t)

	const trigger = "paperless-gpt-scans"
	on := true
	useWorkflows(t, WorkflowConfig{ID: "scans", TriggerTag: trigger, EnableOCR: &on})

	newApp := func(client *recordingClient, processor *mockDocumentProcessor) (*App, *fakeClock) {
		clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
		app := &App{Client: client, docProcessor: processor, ocrProvider: &mockOCRProvider{}, Database: newOCRRunTestDB(t)}
		app.ocrFailures.now = clock.now
		return app, clock
	}
	newClient := func() *recordingClient {
		return &recordingClient{taggedDocuments: map[string][]Document{trigger: {{ID: 41, Tags: []string{trigger}}}}}
	}

	t.Run("unreachable OCR provider stops the round uncounted", func(t *testing.T) {
		client := newClient()
		app, _ := newApp(client, &mockDocumentProcessor{mockErr: errors.New("dial tcp 10.0.0.1:11434: connect: connection refused")})
		_, err := app.processAutoTagDocuments(context.Background())
		assert.ErrorIs(t, err, errServiceUnavailable)
		assert.Empty(t, client.calls)
	})

	t.Run("document failure cools down without global backoff", func(t *testing.T) {
		client := newClient()
		processor := &mockDocumentProcessor{mockErr: errors.New("prediction aborted, token repeat limit reached")}
		app, clock := newApp(client, processor)

		_, err := app.processAutoTagDocuments(context.Background())
		assert.True(t, onlyDocumentErrors(err))
		_, _ = app.processAutoTagDocuments(context.Background())
		assert.Equal(t, 1, processor.ocrCalls[41], "skipped during the cooldown")

		clock.advance(documentRetryBaseDelay)
		_, _ = app.processAutoTagDocuments(context.Background())
		assert.Equal(t, 2, processor.ocrCalls[41])
	})

	t.Run("unreachable paperless after OCR keeps the trigger tag", func(t *testing.T) {
		client := newClient()
		client.updateErr = errors.New("error updating document 41: 503, unavailable")
		client.failAfterNCalls = 1
		app, _ := newApp(client, &mockDocumentProcessor{mockText: "text"})

		_, err := app.processAutoTagDocuments(context.Background())
		assert.ErrorIs(t, err, errServiceUnavailable)
		assert.Len(t, client.calls, 1, "no recovery PATCH")
	})
}

func TestProcessTagDocuments_UnavailableDocumentDetailIsSetAside(t *testing.T) {
	prevAutoTag := autoTag
	t.Cleanup(func() { autoTag = prevAutoTag })
	autoTag = "paperless-gpt-auto"
	isolateWorkflowSettings(t)

	clock := &fakeClock{t: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)}
	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoTag: {{ID: 3, Tags: []string{autoTag}}}},
		getDocumentErr:  errors.New("error fetching document 3: 504, gateway timeout"),
	}
	app := &App{Client: client}
	app.suggestionFailures.now = clock.now

	_, err := app.processAutoTagDocuments(context.Background())
	assert.ErrorIs(t, err, errServiceUnavailable)
	_, waiting := app.suggestionFailures.coolingDown(3)
	assert.True(t, waiting, "the document is set aside so it cannot block the documents behind it")

	_, err = app.processAutoTagDocuments(context.Background())
	assert.NoError(t, err, "the next round skips it")
}

func TestProcessTagDocuments_OCRCooldownDoesNotHoldBackPlainAutoTagging(t *testing.T) {
	prevAutoTag := autoTag
	t.Cleanup(func() { autoTag = prevAutoTag })
	autoTag = "paperless-gpt-auto"
	isolateWorkflowSettings(t)

	client := &recordingClient{
		taggedDocuments: map[string][]Document{autoTag: {{ID: 9, Tags: []string{autoTag}}}},
		getAllTagsErr:   errors.New("exceeds the available context size"),
	}
	app := &App{Client: client}
	app.ocrFailures.recordFailure(9)

	_, err := app.processAutoTagDocuments(context.Background())
	require.Error(t, err, "suggestion generation ran despite the OCR cooldown")
	assert.Contains(t, err.Error(), "document 9")
}

type roundStub struct {
	ocrCount, tagCount int
	ocrErr, tagErr     error
	tagCalls           int
}

func (s *roundStub) isOcrEnabled() bool { return true }
func (s *roundStub) processAutoOcrTagDocuments(context.Context) (int, error) {
	return s.ocrCount, s.ocrErr
}
func (s *roundStub) processAutoTagDocuments(context.Context) (int, error) {
	s.tagCalls++
	return s.tagCount, s.tagErr
}

func TestRunPollRound(t *testing.T) {
	t.Run("document failures in OCR still run auto-tagging and keep counts", func(t *testing.T) {
		stub := &roundStub{ocrCount: 2, ocrErr: documentError{errors.New("pdf is broken")}, tagCount: 1}
		count, err := runPollRound(context.Background(), stub)
		assert.Equal(t, 3, count)
		assert.Equal(t, 1, stub.tagCalls)
		assert.True(t, onlyDocumentErrors(err))
	})

	t.Run("an unreachable service skips auto-tagging", func(t *testing.T) {
		stub := &roundStub{ocrCount: 1, ocrErr: serviceUnavailable(errors.New("connection refused"))}
		count, err := runPollRound(context.Background(), stub)
		assert.Equal(t, 1, count)
		assert.Zero(t, stub.tagCalls)
		assert.ErrorIs(t, err, errServiceUnavailable)
	})

	t.Run("a failure not tied to a document skips auto-tagging", func(t *testing.T) {
		stub := &roundStub{ocrErr: errors.New("error fetching documents with autoOcrTag: 401, unauthorized")}
		_, err := runPollRound(context.Background(), stub)
		assert.Zero(t, stub.tagCalls)
		assert.False(t, onlyDocumentErrors(err))
	})

	t.Run("clean round", func(t *testing.T) {
		stub := &roundStub{ocrCount: 1, tagCount: 1}
		count, err := runPollRound(context.Background(), stub)
		assert.Equal(t, 2, count)
		assert.NoError(t, err)
	})
}
