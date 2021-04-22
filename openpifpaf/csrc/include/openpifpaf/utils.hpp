#pragma once

#define STATIC_GETSET(T, V) static void set_##V(T v) { V = v; }; static T get_##V(void) { return V; };
