/*
 DYNAMICENGINE3D
 AI-Assisted 3D Physics Engine
 By: Elitmers

 Android / arm64 NEON port of the SSE original.
 Every function mirrors its SSE counterpart in DynamicEngine3D/include/MathUtils.h
 lane-for-lane so results stay identical; functions that were scalar in the
 original are left scalar here.
*/

#pragma once

#include "EngineTypes.h"
#include "Constants.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#else
#error "DynamicEngine3DAndroid targets arm64-v8a (NEON) only."
#endif

// Vector3 helpers
inline Vector3 V3(float x, float y, float z) { return { x, y, z, 0.0f }; }
inline Vector3 V3Zero() { return { 0.0f, 0.0f, 0.0f, 0.0f }; }

inline Vector3 Add(const Vector3& a, const Vector3& b) {
    Vector3 r;
    vst1q_f32(&r.x, vaddq_f32(vld1q_f32(&a.x), vld1q_f32(&b.x)));
    return r;
}
inline Vector3 Sub(const Vector3& a, const Vector3& b) {
    Vector3 r;
    vst1q_f32(&r.x, vsubq_f32(vld1q_f32(&a.x), vld1q_f32(&b.x)));
    return r;
}
inline Vector3 Scale(const Vector3& a, float s) {
    Vector3 r;
    vst1q_f32(&r.x, vmulq_n_f32(vld1q_f32(&a.x), s));
    return r;
}
inline Vector3 Neg(const Vector3& a) {
    Vector3 r;
    vst1q_f32(&r.x, vnegq_f32(vld1q_f32(&a.x)));
    return r;
}

// _mm_dp_ps(..., 0x7F) sums x+y+z and EXCLUDES w, so lane 3 is zeroed before the
// reduction. A plain vaddvq_f32 over all four lanes would silently diverge
// wherever w != 0.
inline float Dot(const Vector3& a, const Vector3& b)
{
    float32x4_t p = vmulq_f32(vld1q_f32(&a.x), vld1q_f32(&b.x));
    p = vsetq_lane_f32(0.0f, p, 3);
    return vaddvq_f32(p);
}

// Pairs dot(u,v0) and dot(u,v1) in one pass: x = dot(u,v0), y = dot(u,v1),
// z/w repeat them (the SSE hadd fold broadcasts the same two values).
//
// Note this fold INCLUDES the w product, unlike Dot above — that asymmetry is
// inherited from the original's _mm_hadd_ps and is reproduced deliberately.
inline Vector3 DotPair(const Vector3& u, const Vector3& v0, const Vector3& v1)
{
    float32x4_t uu = vld1q_f32(&u.x);
    const float d0 = vaddvq_f32(vmulq_f32(uu, vld1q_f32(&v0.x)));
    const float d1 = vaddvq_f32(vmulq_f32(uu, vld1q_f32(&v1.x)));
    return { d0, d1, d0, d1 };
}

// Byte-permute tables replacing _mm_shuffle_ps: idxYZX picks lanes (y,z,x,w),
// idxZXY picks (z,x,y,w). Lane N occupies bytes [4N .. 4N+3].
inline Vector3 Cross(const Vector3& a, const Vector3& b)
{
    static const uint8x16_t idxYZX = { 4,5,6,7, 8,9,10,11, 0,1,2,3, 12,13,14,15 };
    static const uint8x16_t idxZXY = { 8,9,10,11, 0,1,2,3, 4,5,6,7, 12,13,14,15 };

    const uint8x16_t va = vreinterpretq_u8_f32(vld1q_f32(&a.x));
    const uint8x16_t vb = vreinterpretq_u8_f32(vld1q_f32(&b.x));

    const float32x4_t a_yzx = vreinterpretq_f32_u8(vqtbl1q_u8(va, idxYZX));
    const float32x4_t b_zxy = vreinterpretq_f32_u8(vqtbl1q_u8(vb, idxZXY));
    const float32x4_t a_zxy = vreinterpretq_f32_u8(vqtbl1q_u8(va, idxZXY));
    const float32x4_t b_yzx = vreinterpretq_f32_u8(vqtbl1q_u8(vb, idxYZX));

    Vector3 r;
    vst1q_f32(&r.x, vsubq_f32(vmulq_f32(a_yzx, b_zxy), vmulq_f32(a_zxy, b_yzx)));
    r.w = 0.0f;
    return r;
}

inline float LengthSq(const Vector3& a) { return Dot(a, a); }
inline float Length(const Vector3& a) { return std::sqrt(Dot(a, a)); }

inline Vector3 Normalize(const Vector3& a)
{
    float len = Length(a);
    if (len < 1e-7f) return { 0.0f, 0.0f, 0.0f, 0.0f };
    float inv = 1.0f / len;
    return { a.x * inv, a.y * inv, a.z * inv, 0.0f };
}

inline Vector3 Lerp(const Vector3& a, const Vector3& b, float t)
{
    return { a.x + (b.x - a.x) * t,
             a.y + (b.y - a.y) * t,
             a.z + (b.z - a.z) * t,
             0.0f };
}

inline float SignF(float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); }
inline float ClampF(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float MaxF(float a, float b) { return a > b ? a : b; }
inline float MinF(float a, float b) { return a < b ? a : b; }
inline float AbsF(float v) { return v < 0.0f ? -v : v; }

// Quaternion helpers
// Kept scalar, exactly as in the SSE original — there is not enough independent
// parallelism in a single quaternion op to pay for the vector setup.
inline Quaternion QIdentity() { return { 0.0f, 0.0f, 0.0f, 1.0f }; }

inline Quaternion QMul(const Quaternion& a, const Quaternion& b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    };
}

inline Quaternion QConj(const Quaternion& q) { return { -q.x, -q.y, -q.z, q.w }; }
inline Quaternion QInverse(const Quaternion& q) { return QConj(q); } // assumes unit quaternion

inline Quaternion QNormalize(const Quaternion& q)
{
    float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-7f) return QIdentity();
    float inv = 1.0f / n;
    return { q.x * inv, q.y * inv, q.z * inv, q.w * inv };
}

inline Vector3 QRotate(const Quaternion& q, const Vector3& v)
{
    // Standard quaternion-vector rotation
    float tx = 2.0f * (q.y * v.z - q.z * v.y);
    float ty = 2.0f * (q.z * v.x - q.x * v.z);
    float tz = 2.0f * (q.x * v.y - q.y * v.x);
    return {
        v.x + q.w * tx + (q.y * tz - q.z * ty),
        v.y + q.w * ty + (q.z * tx - q.x * tz),
        v.z + q.w * tz + (q.x * ty - q.y * tx),
        0.0f
    };
}

// Quaternion from axis-angle (radians)
inline Quaternion QAxisAngle(const Vector3& axis, float angleRad)
{
    float h = angleRad * 0.5f;
    float s = std::sin(h);
    return { axis.x * s, axis.y * s, axis.z * s, std::cos(h) };
}

inline float QDot(const Quaternion& a, const Quaternion& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Shortest-arc slerp matching UnityEngine.Quaternion.Slerp (clamped t, sign-corrected).
inline Quaternion QSlerp(const Quaternion& a, const Quaternion& b, float t)
{
    t = ClampF(t, 0.0f, 1.0f);

    Quaternion end = b;
    float cosTheta = QDot(a, b);
    if (cosTheta < 0.0f)
    {
        end = { -b.x, -b.y, -b.z, -b.w };
        cosTheta = -cosTheta;
    }

    // Near-parallel: lerp and renormalize, which avoids the 1/sin blowup.
    if (cosTheta > 0.9995f)
    {
        return QNormalize(Quaternion{
            a.x + (end.x - a.x) * t,
            a.y + (end.y - a.y) * t,
            a.z + (end.z - a.z) * t,
            a.w + (end.w - a.w) * t });
    }

    const float theta = std::acos(ClampF(cosTheta, -1.0f, 1.0f));
    const float sinTheta = std::sin(theta);
    const float wa = std::sin((1.0f - t) * theta) / sinTheta;
    const float wb = std::sin(t * theta) / sinTheta;

    return QNormalize(Quaternion{
        a.x * wa + end.x * wb,
        a.y * wa + end.y * wb,
        a.z * wa + end.z * wb,
        a.w * wa + end.w * wb });
}

// Left-handed look rotation (Z forward, Y up)

inline Quaternion QLookRotation(const Vector3& forward, const Vector3& up)
{
    const Vector3 z = Normalize(forward);
    if (LengthSq(z) < 0.5f) return QIdentity();

    const Vector3 x = Normalize(Cross(up, z));
    if (LengthSq(x) < 0.5f) return QIdentity();

    const Vector3 y = Cross(z, x);

    // Rotation matrix columns are (x, y, z); convert via Shepperd's method.
    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;

    const float trace = m00 + m11 + m22;
    Quaternion q;

    if (trace > 0.0f)
    {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    }
    else if (m00 > m11 && m00 > m22)
    {
        float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / s;
        q.x = 0.25f * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    }
    else if (m11 > m22)
    {
        float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25f * s;
        q.z = (m12 + m21) / s;
    }
    else
    {
        float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25f * s;
    }

    return QNormalize(q);
}

inline bool IsFinite(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return (bits & 0x7F800000u) != 0x7F800000u;
}

inline bool IsFinite(const Vector3& v)
{
    return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
}

inline bool IsFinite(const Quaternion& q)
{
    return IsFinite(q.x) && IsFinite(q.y) && IsFinite(q.z) && IsFinite(q.w);
}

inline uint32_t SpatialHash(int x, int y, int z) {
    return ((uint32_t)x * 73856093u) ^ ((uint32_t)y * 19349663u) ^ ((uint32_t)z * 83492791u);
}

inline int FloorToInt(float v) { return (int)std::floor(v); }
