#ifndef VECTOR_UTILS_HPP
#define VECTOR_UTILS_HPP

#include <iostream>
#include <map>
#include <queue>

#include "../decoder/field_config.hpp"

void print_vector(vector<float> array);
void print_vector(vector<bool> array);
void print_vector(vector<Seed> array);
void print_vector(vector<pair<int, int>> array);

void print_dims_vector_2d(Vector2D v, string padding="");
void print_dims_vector_3d(Vector3D v);
void print_dims_vector_4d(Vector4D v);

void print_Vector2D(Vector2D v, string padding="");
void print_Vector3D(Vector3D v);
void print_Vector4D(Vector4D v);

void print_Vector2D_nonzeros(Vector2D v);
void print_Vector3D_nonzeros(Vector3D v);

void print_nested_map(map<int, map<int, pair<int, bool>>> by_target);
void print_priority_queue(priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> q);
void print_heap_item(struct HeapItem i);

#endif