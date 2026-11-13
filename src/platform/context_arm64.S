#if defined(__aarch64__)

// The context switch on arm64. A saved context is a 160-byte frame on its
// stack: x19-x28, the frame pointer x29, the link register x30 (at byte 88)
// and d8-d15, the registers that the calling convention keeps across a call.

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

        .text
        .p2align 2
        .globl SYMBOL(context_switch)
SYMBOL(context_switch):
        sub sp, sp, #160
        stp x19, x20, [sp, #0]
        stp x21, x22, [sp, #16]
        stp x23, x24, [sp, #32]
        stp x25, x26, [sp, #48]
        stp x27, x28, [sp, #64]
        stp x29, x30, [sp, #80]
        stp d8, d9, [sp, #96]
        stp d10, d11, [sp, #112]
        stp d12, d13, [sp, #128]
        stp d14, d15, [sp, #144]
        mov x9, sp
        str x9, [x0]
        mov sp, x1
        ldp x19, x20, [sp, #0]
        ldp x21, x22, [sp, #16]
        ldp x23, x24, [sp, #32]
        ldp x25, x26, [sp, #48]
        ldp x27, x28, [sp, #64]
        ldp x29, x30, [sp, #80]
        ldp d8, d9, [sp, #96]
        ldp d10, d11, [sp, #112]
        ldp d12, d13, [sp, #128]
        ldp d14, d15, [sp, #144]
        add sp, sp, #160
        ret

// A new context: a zeroed frame below top whose x30 is the entry, so that the
// first switch to it returns into the entry.
        .p2align 2
        .globl SYMBOL(context_new)
SYMBOL(context_new):
        sub x0, x0, #160
        stp xzr, xzr, [x0, #0]
        stp xzr, xzr, [x0, #16]
        stp xzr, xzr, [x0, #32]
        stp xzr, xzr, [x0, #48]
        stp xzr, xzr, [x0, #64]
        stp xzr, x1, [x0, #80]
        stp xzr, xzr, [x0, #96]
        stp xzr, xzr, [x0, #112]
        stp xzr, xzr, [x0, #128]
        stp xzr, xzr, [x0, #144]
        ret
#endif
