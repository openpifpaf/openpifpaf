#include <cmath>

#include "../../include/utils/numpy_utils.hpp"


float maximum1d(const vector<float> v) {
    float max = v[0];
    for(int i = 0; i < v.size(); i++) {
        if(v[i] > max)
            max = v[i];
    }
    return max;
}

float minimum1d(const vector<float> v) {
    float min_ = v[0];
    for(int i = 0; i < v.size(); i++) {
        if(v[i] < min_)
            min_ = v[i];
    }
    return min_;
}

float norm2d(const Vector2D &v) {
    int d1 = v.size();
    int d2 = v[0].size();

    float r = 0;
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            float a = abs(v[i][j]);
            r += a * a;
        }
    }

    return sqrt(r);
}

vector<float> norm2d_axis0(const Vector2D &v) {
    int d1 = v.size();
    int d2 = v[0].size();

    vector<float> r;
    for(int i = 0; i < d2; i++) {
        float s = 0;
        for(int j = 0; j < d1; j++) {
            float a = abs(v[j][i]);
            s += a * a;
        }
        r.push_back(sqrt(s));
    }

    return r;
}


vector<float> full1d(int size, float val) {
    vector<float> r;
    for(int i = 0; i < size; i ++) {
        r.push_back(val);
    }

    return r;
}

vector<bool> mask1d(const vector<float> &v, float threshold) {
    vector<bool> r;
    for(int i = 0; i < v.size(); i++) {
        if(v[i] > threshold) {
            r.push_back(true);
        }
        else {
            r.push_back(false);
        }
    }

    return r;
}

vector<bool> mask1d_lt(const vector<float> &v, float threshold) {
    vector<bool> r;
    for(int i = 0; i < v.size(); i++) {
        if(v[i] < threshold) {
            r.push_back(true);
        }
        else {
            r.push_back(false);
        }
    }

    return r;
}

blobnd<bool> mask2d(
    blobnd<float> &v, int index_d4, int index_d3,
    float filter_value
) {
    int d1 = v.shape[2];
    int d2 = v.shape[3];

    blobnd<bool> mask({static_cast<unsigned int>(d1), static_cast<unsigned int>(d2)});
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            if(v(index_d4, index_d3, i, j) > filter_value)
                mask(i, j) = 1;
        }
    }

    return mask;
}

Vector2D filter_vector3D(
    blobnd<float> &v, int index_d4, int filter_index,
    blobnd<bool> &mask
) {
    unsigned d1 = v.shape[0];
    unsigned d2 = v.shape[1];
    unsigned d3 = v.shape[2];
    unsigned d4 = v.shape[3];

    Vector2D result;
    for(unsigned i = 0; i < d2; i++) {
        vector<float> sample_result;
        for(unsigned j = 0; j < d3; j++) {
            for(unsigned k = 0; k < d4; k++) {
                if(mask(j, k))
                     sample_result.push_back(v(index_d4, i, j, k));
            }
        }
        result.push_back(sample_result);
    }

    return result;
}

vector<float> filter1d(const vector<float> &v, const vector<bool> &mask) {
    vector<float> r;
    for(int i = 0; i < v.size(); i++) {
        if(mask[i]) {
            r.push_back(v[i]);
        }
    }

    return r;
}

vector<float> maximum(float threshold, const vector<float> &array) {
    /*
        Equivalent for python numpy np.maximum for 1D usecase

        Example:
        array = [
            0.8964476,  0.9076828,  0.901904,   0.89791566, 0.88608307, 0.90584385,
            0.90705764, 0.9029248,  0.8968029,  0.8863319,  1.0112386,  1.0113224,
            0.9996567,  1.0156801,  1.0094,     1.023006,   1.0241847,  1.0096267,
            1.0292764,  1.0121837
        ]

        r = np.maximum(1.0, array)
        r = [
            1., 1., 1., 1., 1., 1.
            1., 1., 1., 1., 1.0112386, 1.0113224,
            1., 1.0156801, 1.0094, 1.023006, 1.0241847, 1.0096267,
            1.0292764, 1.0121837
        ]

    */
    vector<float> result;

    for(int i = 0; i < array.size(); i++) {
        if(array[i] < threshold) {
            result.push_back(threshold);
        } else {
            result.push_back(array[i]);
        }
    }

    return result;
}




Vector2D filter_vector2D(const Vector2D &v, const vector<bool> &mask) {
    int d1 = v.size();
    int d2 = v[0].size();

    Vector2D r;
    for(int i = 0; i < d1; i++) {
        vector<float> v1;
        for(int j = 0; j < d2; j++) {
            if(mask[j]) {
                v1.push_back(v[i][j]);
            }
        }
        r.push_back(v1);
    }

    return r;
}

Vector2D filter_vector2D(const Vector2D &v, int filter_index, float filter_value) {
    vector<bool> mask;
    vector<float> target_vector = v[filter_index];
    for(int i = 0; i < target_vector.size(); i++) {
        if(target_vector[i] > filter_value) {
            mask.push_back(true);
        } else {
            mask.push_back(false);
        }
    }

    int d1 = v.size();
    int d2 = v[0].size();

    Vector2D r;
    for(int i = 0; i < d1; i++) {
        vector<float> v1;
        for(int j = 0; j < d2; j++) {
            if(mask[j]) {
                v1.push_back(v[i][j]);
            }
        }
        r.push_back(v1);
    }

    return r;
}

Vector3D maximum3d(const Vector3D &a, const Vector3D &b) {
    int d1 = a.size();
    int d2 = a[0].size();
    int d3 = a[0][0].size();

    Vector3D c;
    Vector2D t;
    vector<float> v;
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                if(a[i][j][k] > b[i][j][k]) {
                    v.push_back(a[i][j][k]);
                } else {
                    v.push_back(b[i][j][k]);
                }
            }
            t.push_back(v);
        }
       c.push_back(t);
    }

    return c;
}

vector<float> vector1d_ones(int shape) {
    vector<float> r;
    for(int i = 0; i < shape; i++) {
        r.push_back(1.0);
    }

    return r;
}

vector<float> vector1d_zeros(int shape) {
    vector<float> r;
    for(int i = 0; i < shape; i++) {
        r.push_back(0);
    }

    return r;
}

Vector2D vector2d_zeros(const vector<int> &shape) {
    int d1 = shape[0];
    int d2 = shape[1];

    Vector2D v2;
    vector<float> v1;
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            v1.push_back(0.0);
        }
        v2.push_back(v1);
        v1.clear();
    }

    return v2;
}

Vector3D vector3d_zeros(const vector<int> &shape) {
    int d1 = shape[0];
    int d2 = shape[1];
    int d3 = shape[2];

    vector<float> d3_0(d3, 0);
    vector<vector<float>> d2_0(d2, d3_0);
    vector<vector<vector<float>>> v3(d1, d2_0);

    return v3;
}

Vector4D vector4d_zeros(const vector<int> &shape) {
    int d1 = shape[0];
    int d2 = shape[1];
    int d3 = shape[2];
    int d4 = shape[3];

    Vector4D v4;
    Vector3D v3;
    Vector2D v2;
    vector<float> v1;
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                for(int l = 0; l < d4; l++) {
                    v1.push_back(0.0);
                }
                v2.push_back(v1);
                v1.clear();
            }
            v3.push_back(v2);
            v2.clear();
        }
        v4.push_back(v3);
        v3.clear();
    }

    return v4;
}

Vector2D subtract2d(const Vector2D &v1, const Vector2D &v2) {
    int d1 = v1.size();
    int d2 = v1[0].size();
    Vector2D r;

    vector<float> r1;
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            float c = v1[i][j] - v2[i][j];
            r1.push_back(c);
        }
        r.push_back(r1);
        r1.clear();
    }

    return r;
}


