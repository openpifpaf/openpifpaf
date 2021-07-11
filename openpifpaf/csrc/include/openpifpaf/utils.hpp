#pragma once


#ifdef _WIN32
#if defined(OPENPIFPAF_DLLEXPORT)
#define OPENPIFPAF_API __declspec(dllexport)
#else
#define OPENPIFPAF_API __declspec(dllimport)
#endif
#else
#define OPENPIFPAF_API
#endif
