package main

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

func apiError(statusCode int) error {
	return &openai.Error{StatusCode: statusCode}
}

func functionCallResponse(callID string) *responses.Response {
	return &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type:      "function_call",
				CallID:    callID,
				Name:      "unknown_tool",
				Arguments: responses.ResponseOutputItemUnionArguments{OfString: "{}"},
			},
		},
	}
}

func TestCallWithRetryRetriesTransientErrorOnce(t *testing.T) {
	originalDelay := retryDelay
	retryDelay = time.Millisecond
	t.Cleanup(func() { retryDelay = originalDelay })

	attempts := 0
	callAPI := func(context.Context, responses.ResponseInputParam, []responses.ToolUnionParam) (*responses.Response, error) {
		attempts++
		if attempts == 1 {
			return nil, apiError(429)
		}
		return &responses.Response{}, nil
	}

	resp, err := callWithRetry(context.Background(), callAPI, nil, nil)
	if err != nil {
		t.Fatalf("callWithRetry() error = %v", err)
	}
	if resp == nil {
		t.Fatal("callWithRetry() returned a nil response")
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2 (original plus one retry)", attempts)
	}
}

func TestCallWithRetryDoesNotRetryClientError(t *testing.T) {
	attempts := 0
	callAPI := func(context.Context, responses.ResponseInputParam, []responses.ToolUnionParam) (*responses.Response, error) {
		attempts++
		return nil, apiError(400)
	}

	if _, err := callWithRetry(context.Background(), callAPI, nil, nil); err == nil {
		t.Fatal("callWithRetry() error = nil, want HTTP 400 error")
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func TestCallWithRetryStopsAfterOneRetry(t *testing.T) {
	originalDelay := retryDelay
	retryDelay = time.Millisecond
	t.Cleanup(func() { retryDelay = originalDelay })

	attempts := 0
	callAPI := func(context.Context, responses.ResponseInputParam, []responses.ToolUnionParam) (*responses.Response, error) {
		attempts++
		return nil, apiError(503)
	}

	_, err := callWithRetry(context.Background(), callAPI, nil, nil)
	if err == nil {
		t.Fatal("callWithRetry() error = nil, want exhausted retry error")
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
}

func TestCallWithRetryRespectsContextDeadlineWhileWaitingToRetry(t *testing.T) {
	originalDelay := retryDelay
	retryDelay = 50 * time.Millisecond
	t.Cleanup(func() { retryDelay = originalDelay })

	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	callAPI := func(context.Context, responses.ResponseInputParam, []responses.ToolUnionParam) (*responses.Response, error) {
		return nil, apiError(500)
	}

	_, err := callWithRetry(ctx, callAPI, nil, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("callWithRetry() error = %v, want context.DeadlineExceeded", err)
	}
}

func TestAgentLoopReturnsMaxStepsError(t *testing.T) {
	attempts := 0
	callAPI := func(context.Context, responses.ResponseInputParam, []responses.ToolUnionParam) (*responses.Response, error) {
		attempts++
		return functionCallResponse(fmt.Sprintf("call-%d", attempts)), nil
	}

	err := agent_loop(context.Background(), callAPI)
	if !errors.Is(err, ErrMaxSteps) {
		t.Fatalf("agent_loop() error = %v, want ErrMaxSteps", err)
	}
	if attempts != maxSteps {
		t.Fatalf("attempts = %d, want %d", attempts, maxSteps)
	}
}
