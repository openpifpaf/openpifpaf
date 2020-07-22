#ifndef SAMPLE_DATA_HPP
#define SAMPLE_DATA_HPP

constexpr int inputWidth = 129;
constexpr int inputHeight = 97;
constexpr int tensorWidth = 17;
constexpr int tensorHeight = 13;
constexpr int tensorSize = tensorWidth * tensorHeight;
extern const float pif_c[17 * tensorSize];
extern const float pif_r[17 * 2 * tensorSize];
extern const float pif_b[17 * tensorSize];
extern const float pif_s[17 * tensorSize];
extern const float paf_c[19 * tensorSize];
extern const float paf_r1[19 * 2 * tensorSize];
extern const float paf_r2[19 * 2 * tensorSize];
extern const float paf_b1[19 * tensorSize];
extern const float paf_b2[19 * tensorSize];

#endif // SAMPLE_DATA_HPP
