#include <cmath>

#include "../../include/decoder/functional.hpp"
#include "../../include/utils/numpy_utils.hpp"


float clip(float a, float min_value, float max_value) {
    // return min(max(a, min_value), max_value);
    return max(min_value, min(max_value, a));
}

float approx_exp(float x) {
    if(x > 2.0 || x < -2.0) {
        return 0.0;
    }

    x = 1.0 + x / 8.0;
    x *= x;
    x *= x;
    x *= x;
    return x;
}

void scalar_square_add_gauss_with_max(
    blobnd<float> &field, int field_index,
    const vector<float> &x,
    const vector<float> &y,
    const vector<float> &sigma,
    const vector<float> &v,
    float truncate,
    float max_value
) {
    float vv, deltax2, deltay2;
    float cv, cx, cy, csigma, csigma2;
    float truncate2 = truncate * truncate;
    float truncate_csigma, truncate2_csigma2;

    int i, xx, yy;
    long minx = 0;
    long miny = 0;
    long maxx = 0;
    long maxy = 0;

    int field_rows = field.shape[1];
    int field_columns = field.shape[2];

    for(i = 0; i < x.size(); i++) {
        csigma = sigma[i];
        csigma2 = csigma * csigma;
        truncate_csigma = truncate * csigma;
        truncate2_csigma2 = truncate2 * csigma2;

        cx = x[i];
        cy = y[i];
        cv = v[i];

        minx = (long)(clip(cx - truncate_csigma, 0.0, field_columns - 1));
        maxx = (long)(clip(cx + truncate_csigma + 1, minx + 1.0, field_columns));
        miny = (long)(clip(cy - truncate_csigma, 0.0, field_rows - 1));
        maxy = (long)(clip(cy + truncate_csigma + 1, miny + 1.0, field_rows));

        for(xx = minx; xx < maxx; xx++) {
            deltax2 = (xx - cx) * (xx - cx);
            for(yy = miny; yy < maxy; yy++) {
                deltay2 = (yy - cy) * (yy - cy);

                if(deltax2 + deltay2 > truncate2_csigma2) {
                    continue;
                }
                if(deltax2 < 0.25 && deltay2 < 0.25) {
                    vv = cv;
                } else {
                    vv = cv * approx_exp(-0.5 * (deltax2 + deltay2) / csigma2);
                }

                field(field_index, yy, xx) = field(field_index, yy, xx) + vv;
                field(field_index, yy, xx) = min(max_value, field(field_index, yy, xx));
            }
        }
    }
}

vector<float> scalar_values(
    blobnd<float> &field, int field_index,
    const vector<float> &x,
    const vector<float> &y,
    float default_val,
    int debug
) {
    int d1 = field.shape[1];
    int d2 = field.shape[2];
    vector<float> values = full1d(x.size(), default_val);
    float maxx = (float)(d2 - 0.51);
    float maxy = (float)(d1 - 0.51);

    for(int i = 0; i < values.size(); i++) {
        if(x[i] < -0.49 || y[i] < -0.49 || x[i] > maxx || y[i] > maxy) {
            continue;
        }

        values[i] = field(field_index, (int)(y[i]+0.5), (int)(x[i]+0.5));
    }

    return values;
}

float scalar_nonzero_clipped_with_reduction(const Vector2D &field, float x, float y, float r) {
    int d1 = field.size();
    int d2 = field[0].size();

    float xc = clip(x / r, 0.0, d2 - 1);
    float yc = clip(y / r, 0.0, d1 - 1);

    return field[ (int)yc ][ (int)xc ];
}

void scalar_square_add_single(Vector2D &field, float x, float y, float sigma, float value) {
    int d1 = field.size();
    int d2 = field[0].size();

    int minx = max(0, (int)(x - sigma));
    int miny = max(0, (int)(y - sigma));
    int maxx = max(minx + 1, min(d2, (int)(x + sigma) + 1));
    int maxy = max(miny + 1, min(d1, (int)(y + sigma) + 1));

    for(int i = miny; i < maxy; i++) {
        for(int j = minx; j < maxx; j++) {
            field[i][j] += value;
        }
    }
}

vector<float> grow_connection_blend(const Vector2D &caf_field, float x, float y, float xy_scale, bool only_max) {
    float sigma_filter = 2.0 * xy_scale;
    float sigma2 = 0.25 * xy_scale * xy_scale;
    float d2 = 0.0;
    float v = 0.0;
    float score = 0.0;

    unsigned score_1_i = 0;
    unsigned score_2_i = 0;
    float score_1 = 0.0;
    float score_2 = 0.0;

    for(int i = 0; i < caf_field[0].size(); i++) {
        if(caf_field[1][i] < x - sigma_filter)
            continue;
        if(caf_field[1][i] > x + sigma_filter)
            continue;
        if(caf_field[2][i] < y - sigma_filter)
            continue;
        if(caf_field[2][i] > y + sigma_filter)
            continue;

        float p1 = (caf_field[1][i] - x) * (caf_field[1][i] - x);
        float p2 = (caf_field[2][i] - y) * (caf_field[2][i] - y);
        d2 = p1 + p2;

        score = exp(-0.5 * d2 / sigma2) * caf_field[0][i];

        if(score >= score_1) {
            score_2_i = score_1_i;
            score_2 = score_1;
            score_1_i = i;
            score_1 = score;
        } else if(score > score_2) {
            score_2_i = i;
            score_2 = score;
        }
    }

    if(score_1 == 0.0)
        return vector<float> {0.0, 0.0, 0.0, 0.0};

    // only max
    vector<float> entry_1 = {
        caf_field[3][score_1_i], caf_field[4][score_1_i],
        caf_field[6][score_1_i], caf_field[8][score_1_i]
    };
    if(only_max)
       return vector<float> {entry_1[0], entry_1[1], entry_1[3], score_1};

    // blend
    vector<float> entry_2 = {
        caf_field[3][score_2_i], caf_field[4][score_2_i],
        caf_field[6][score_2_i], caf_field[8][score_2_i]
    };
    if(score_2 < 0.01 || score_2 < 0.5 * score_1) {
        return vector<float> {entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5f};
    }

    float blend_d2 = ((entry_1[0] - entry_2[0]) * (entry_1[0] - entry_2[0])) + ((entry_1[1] - entry_2[1]) * (entry_1[1] - entry_2[1]));
    if(blend_d2 > (entry_1[3] * entry_1[3]) / 4.0) {
        return vector<float> {entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5f};
    }

    return vector<float> {
        (score_1 * entry_1[0] + score_2 * entry_2[0]) / (score_1 + score_2),
        (score_1 * entry_1[1] + score_2 * entry_2[1]) / (score_1 + score_2),
        (score_1 * entry_1[3] + score_2 * entry_2[3]) / (score_1 + score_2),
        0.5f * (score_1 + score_2),
    };
}
