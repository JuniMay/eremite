#pragma once

// Stop requests, from SIGINT, SIGTERM or the program itself.

// Watches SIGINT and SIGTERM: makes them request a stop. The handlers count
// the request and write one byte to a pipe, and do no other I/O. After each
// request, hook runs on a thread of its own, if it is not null. The first hook
// stays.
void shutdown_watch(void (*hook)(void));

// Requests a stop, as SIGINT and SIGTERM do.
void shutdown_request(void);

// Whether a stop was requested.
int shutdown_requested(void);

// How many stops were requested.
int shutdown_requests(void);
