// zlib shim — provides zlib-compatible symbols backed by miniz.
//
// PDFium is compiled against our standalone zlib.h (correct zlib types).
// At link time, these implementations forward to miniz functions.
// The z_stream and mz_stream structs have identical layouts on 64-bit.

#include "miniz.h"
#include <string.h>

// zlib types (matching our zlib.h)
typedef unsigned char Byte;
typedef unsigned int uInt;
typedef unsigned long uLong;
typedef Byte Bytef;
typedef uLong uLongf;
typedef void *voidpf;

typedef voidpf (*alloc_func_zlib)(voidpf opaque, uInt items, uInt size);
typedef void   (*free_func_zlib)(voidpf opaque, voidpf address);

struct internal_state;

typedef struct z_stream_s {
    const Bytef *next_in;
    uInt     avail_in;
    uLong    total_in;
    Bytef    *next_out;
    uInt     avail_out;
    uLong    total_out;
    const char *msg;
    struct internal_state *state;
    alloc_func_zlib zalloc;
    free_func_zlib  zfree;
    voidpf     opaque;
    int     data_type;
    uLong   adler;
    uLong   reserved;
} z_stream;

typedef z_stream *z_streamp;

// Convert between z_stream and mz_stream.
// Layouts are compatible: same field order, next_in/next_out/msg pointers,
// avail_in/avail_out as unsigned int, total_in/total_out/adler/reserved as uLong.
// Only difference: alloc/free function pointer types (unsigned int vs size_t params).
// We copy fields individually to avoid strict aliasing issues.
// We always provide miniz's default allocators because miniz_def_free_func is
// called unconditionally in mz_inflateEnd/mz_deflateEnd to free internal state.
static void z_to_mz(const z_stream *z, mz_stream *m) {
    m->next_in = z->next_in;
    m->avail_in = z->avail_in;
    m->total_in = z->total_in;
    m->next_out = z->next_out;
    m->avail_out = z->avail_out;
    m->total_out = z->total_out;
    m->msg = z->msg;
    m->state = (struct mz_internal_state *)z->state;
    m->zalloc = miniz_def_alloc_func;
    m->zfree = miniz_def_free_func;
    m->opaque = z->opaque;
    m->data_type = z->data_type;
    m->adler = z->adler;
    m->reserved = z->reserved;
}

static void mz_to_z(const mz_stream *m, z_stream *z) {
    z->next_in = m->next_in;
    z->avail_in = m->avail_in;
    z->total_in = m->total_in;
    z->next_out = m->next_out;
    z->avail_out = m->avail_out;
    z->total_out = m->total_out;
    z->msg = m->msg;
    z->state = (struct internal_state *)m->state;
    // zalloc/zfree not copied back
    z->data_type = m->data_type;
    z->adler = m->adler;
    z->reserved = m->reserved;
}

int inflateInit_(z_streamp strm, const char *version, int stream_size) {
    (void)version;
    (void)stream_size;
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_inflateInit(&ms);
    mz_to_z(&ms, strm);
    return ret;
}

int inflateInit2_(z_streamp strm, int windowBits, const char *version, int stream_size) {
    (void)version;
    (void)stream_size;
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_inflateInit2(&ms, windowBits);
    mz_to_z(&ms, strm);
    return ret;
}

int inflate(z_streamp strm, int flush) {
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_inflate(&ms, flush);
    mz_to_z(&ms, strm);
    return ret;
}

int inflateEnd(z_streamp strm) {
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_inflateEnd(&ms);
    mz_to_z(&ms, strm);
    return ret;
}

int deflateInit_(z_streamp strm, int level, const char *version, int stream_size) {
    (void)version;
    (void)stream_size;
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_deflateInit(&ms, level);
    mz_to_z(&ms, strm);
    return ret;
}

int deflateInit2_(z_streamp strm, int level, int method, int windowBits,
                  int memLevel, int strategy, const char *version, int stream_size) {
    (void)version;
    (void)stream_size;
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_deflateInit2(&ms, level, method, windowBits, memLevel, strategy);
    mz_to_z(&ms, strm);
    return ret;
}

int deflate(z_streamp strm, int flush) {
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_deflate(&ms, flush);
    mz_to_z(&ms, strm);
    return ret;
}

int deflateEnd(z_streamp strm) {
    mz_stream ms;
    memset(&ms, 0, sizeof(ms));
    z_to_mz(strm, &ms);
    int ret = mz_deflateEnd(&ms);
    mz_to_z(&ms, strm);
    return ret;
}

int compress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen) {
    mz_ulong mz_destLen = *destLen;
    int ret = mz_compress(dest, &mz_destLen, source, (mz_ulong)sourceLen);
    *destLen = mz_destLen;
    return ret;
}

int compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level) {
    mz_ulong mz_destLen = *destLen;
    int ret = mz_compress2(dest, &mz_destLen, source, (mz_ulong)sourceLen, level);
    *destLen = mz_destLen;
    return ret;
}

int uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen) {
    mz_ulong mz_destLen = *destLen;
    int ret = mz_uncompress(dest, &mz_destLen, source, (mz_ulong)sourceLen);
    *destLen = mz_destLen;
    return ret;
}

uLong compressBound(uLong sourceLen) {
    return (uLong)mz_compressBound((mz_ulong)sourceLen);
}

uLong adler32(uLong adler, const Bytef *buf, uInt len) {
    return (uLong)mz_adler32((mz_ulong)adler, buf, len);
}

uLong crc32(uLong crc, const Bytef *buf, uInt len) {
    return (uLong)mz_crc32((mz_ulong)crc, buf, len);
}

const char *zlibVersion(void) {
    return "1.3.1";
}
