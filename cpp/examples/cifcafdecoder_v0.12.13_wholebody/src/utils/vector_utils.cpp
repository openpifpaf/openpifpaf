#include "../../include/utils/vector_utils.hpp"

#include <iostream>

void print_vector(vector<float> array) {
    cout << "[ ";
    for(int i = 0; i < array.size(); i++) {
        cout << array[i] << " ";
    }
    cout << "] ";
    cout << endl;

}

void print_vector(vector<bool> array) {
    cout << "[ ";
    for(int i = 0; i < array.size(); i++) {
        cout << array[i] << " ";
    }
    cout << "] ";
    cout << endl;

}

void print_vector(vector<pair<int, int>> array) {
    cout << "[ ";
    for(int i = 0; i < array.size(); i++) {
        cout << "(" << array[i].first << ", " << array[i].second << ") ";
    }
    cout << "] ";
    cout << endl;
}

void print_vector(vector<Seed> array) {
    cout << "[ ";
    for(int i = 0; i < array.size(); i++) {
        struct Seed s = array[i];
        cout << "(" << s.vv << ", " << s.field << ", " << s.xx << ", " << s.yy << ", " << s.ss << ")";
    }
    cout << "] ";
    cout << endl;
}

void print_dims_vector_2d(Vector2D v, string padding) {
    int d1 = v.size();
    int d2 = v[0].size();

    cout << padding << "(" << d1 << ", " << d2 << ")" << endl;
}

void print_dims_vector_3d(Vector3D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    int d3 = v[0][0].size();

    cout << "(" << d1 << ", " << d2 << ", " << d3 << ")" << endl;
}

void print_dims_vector_4d(Vector4D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    int d3 = v[0][0].size();
    int d4 = v[0][0][0].size();

    cout << "(" << d1 << ", " << d2 << ", " << d3 << ", " << d4 << ")" << endl;
}

void print_Vector2D(Vector2D v, string padding) {
    int rows = v.size();
    int columns = v[0].size();

    print_dims_vector_2d(v, padding);

    cout << padding << "[" << endl;
    for(int i = 0; i < rows; i++) {
        if(i == 3) {
            cout << padding << "  ..." << endl;
            continue;
        } else if (i <= 2 || i >= rows - 3) {
            for(int j = 0; j < columns; j++) {
                if(columns > 6) {
                    if (j <= 2 || j >= columns - 3) {
                        cout << padding << "  " << v[i][j] << " ";
                    } else if (j == 3) {
                        cout << padding << " ... ";
                    }
                } else {
                    cout << padding << v[i][j] << ' ';
                }
            }
            cout << endl;
        }
    }
    cout << padding << "]" << endl;
}

void print_Vector3D(Vector3D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    int d3 = v[0][0].size();

    print_dims_vector_3d(v);
    cout << "[" << endl;
    for(int i = 0; i < d1; i++) {
        if(i == 3 || i == 4 || i == 5)
            cout << "  ." << endl;
        else if(i <= 2 || i >= d1 - 3)
            print_Vector2D(v[i], "  ");
        else
            continue;
    }
    cout << "]" << endl;
}

void print_Vector4D(Vector4D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    int d3 = v[0][0].size();
    int d4 = v[0][0][0].size();

    cout << "[" << endl;
    for(int i = 0; i < d1; i++) {
        cout << " [" << endl;
        for(int j = 0; j < d2; j++) {
            cout << "  [" << endl;
            for(int k = 0; k < d3; k++) {
                for(int l = 0; l < d4; l++) {
                    cout << "  " << v[i][j][k][l] << " ";
                }
                cout << endl;
            }
            cout << "  ]" << endl;
        }
        cout << " ]" << endl;
    }
    cout << "]" << endl;
}

void print_Vector3D_nonzeros(Vector3D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    int d3 = v[0][0].size();

    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                if(v[i][j][k] != 0.0)
                    // cout << i << j << k << endl;
                    cout << v[i][j][k] << endl;
            }
        }
    }
}

void print_Vector2D_nonzeros(Vector2D v) {
    int d1 = v.size();
    int d2 = v[0].size();
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            if(v[i][j] != 0.0)
                // cout << i << j << k << endl;
                cout << v[i][j] << endl;
        }

    }
}

void print_nested_map(map<int, map<int, pair<int, bool>>> by_target) {
    map<int, map<int, pair<int, bool>>>::iterator it;
    map<int, pair<int, bool>>::iterator jt;
    for (it = by_target.begin(); it != by_target.end(); it++) {
        int key_it = it->first;
        map<int, pair<int, bool>> val_it = it->second;
        cout << "(" << key_it << ", {";
        for(jt = val_it.begin(); jt != val_it.end(); jt++) {
            int key_jt = jt->first;
            pair<int, bool> val_jt = jt->second;
            cout << key_jt << ": (" << val_jt.first << ", " << val_jt.second << "), ";
        }
        cout << " })" << endl;
    }
}

void print_priority_queue(priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> q) {
    cout << "[";

    while (!q.empty()) {
        struct HeapItem i =  q.top();
        cout << "(" << i.score << ", ";

        cout << "[";
        for(int j = 0; j < i.new_xysv.size(); j++) {
            cout << i.new_xysv[j] << ", ";
        }
        cout << "], ";

        cout << i.start_i << ", " << i.end_i << "), ";
        q.pop();
    }
    cout << "]" << endl;
}

void print_heap_item(struct HeapItem i) {
    cout << "(" << i.score << ", ";

    cout << "[";
    for(int j = 0; j < i.new_xysv.size(); j++) {
        cout << i.new_xysv[j] << ", ";
    }
    cout << "], ";

    cout << i.start_i << ", " << i.end_i << ")" << endl;

}