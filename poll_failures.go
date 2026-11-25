package main

import (
	"errors"
	"io"
	"net"
	"net/url"
	"regexp"
	"strings"
	"syscall"
	"time"
)

// errServiceUnavailable marks a poll round that stopped because paperless-ngx
// or an LLM/OCR provider could not be reached. Such a failure is not held
// against the document that happened to be in flight; the whole poll loop
// backs off instead.
var errServiceUnavailable = errors.New("service unavailable")

var connectionErrorFragments = []string{
	"connection refused",
	"connection reset by peer",
	"broken pipe",
	"no such host",
	"network is unreachable",
	"no route to host",
	"dial tcp",
	"tls handshake timeout",
	"server misbehaving",
	"server busy", // Ollama's 503 body when its request queue is full
	"code = unavailable",
	"code = resourceexhausted",
}

// Matches 429/502/503/504 in the formats the LLM, OCR and paperless-ngx
// clients report them. A 500 is deliberately absent: the server got the
// request and failed on it, which can be caused by the document itself.
var unavailableStatusPattern = regexp.MustCompile(`status code:? (429|502|503|504)\b|returned status (429|502|503|504)\b|status: (429|502|503|504)\b|: (429|502|503|504), |status=(429|502|503|504)\b`)

// An EOF counts only as a broken HTTP connection (`Post "url": EOF`); a bare
// "unexpected EOF" also comes from decoding a truncated file.
var httpEOFPattern = regexp.MustCompile(`(get|post|put|patch|delete|head) "[^"]*": (unexpected )?eof\b`)

// isServiceUnavailable reports whether err means a service could not be
// reached or refused to serve right now, as opposed to failing on the
// document. Timeouts after a connection was made are document failures: a
// page that always takes too long must still count toward the retry limit.
func isServiceUnavailable(err error) bool {
	if err == nil {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return true
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	for _, target := range []error{syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.EPIPE, syscall.EHOSTUNREACH, syscall.ENETUNREACH} {
		if errors.Is(err, target) {
			return true
		}
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) && (errors.Is(urlErr.Err, io.EOF) || errors.Is(urlErr.Err, io.ErrUnexpectedEOF)) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, fragment := range connectionErrorFragments {
		if strings.Contains(msg, fragment) {
			return true
		}
	}
	return unavailableStatusPattern.MatchString(msg) || httpEOFPattern.MatchString(msg)
}

// documentError is a failure attributed to a single document. It is handled
// by that document's retry cooldown and retry limit, so it never puts the
// whole poll loop in backoff.
type documentError struct{ err error }

func (e documentError) Error() string { return e.err.Error() }
func (e documentError) Unwrap() error { return e.err }

// onlyDocumentErrors reports whether every failure inside err (following
// fmt.Errorf %w wrapping and errors.Join) is a documentError.
func onlyDocumentErrors(err error) bool {
	switch e := err.(type) {
	case nil:
		return false
	case documentError:
		return true
	case interface{ Unwrap() []error }:
		children := e.Unwrap()
		if len(children) == 0 {
			return false
		}
		for _, child := range children {
			if !onlyDocumentErrors(child) {
				return false
			}
		}
		return true
	case interface{ Unwrap() error }:
		return onlyDocumentErrors(e.Unwrap())
	}
	return false
}

// pollBackoff decides how long the background loop waits before its next
// round. Rounds whose only failures are documentErrors poll at the normal
// pace; any other failure (an unreachable service, a failing listing call)
// waits exponentially longer, up to max, until a round gets through.
type pollBackoff struct {
	min, max, interval time.Duration
	current            time.Duration
}

func (b *pollBackoff) next(processed int, err error) time.Duration {
	if err != nil && !onlyDocumentErrors(err) {
		if b.current == 0 {
			b.current = b.min
		} else {
			b.current *= 2
		}
		if b.current > b.max {
			b.current = b.max
		}
		return b.current
	}
	b.current = 0
	if processed == 0 {
		return b.interval
	}
	return 0
}
