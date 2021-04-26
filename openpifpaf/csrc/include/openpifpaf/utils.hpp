#pragma once

// TODO the following is a temporary workaround for https://github.com/pytorch/pytorch/issues/56571
#define STATIC_GETSET(T, V) static void set_##V(T v) { V = v; }; static T get_##V(void) { return V; };


#ifdef _WIN32
#if defined(OPENPIFPAF_DLLEXPORT)
#define OPENPIFPAF_API __declspec(dllexport)
#else
#define OPENPIFPAF_API __declspec(dllimport)
#endif
#else
#define OPENPIFPAF_API
#endif
