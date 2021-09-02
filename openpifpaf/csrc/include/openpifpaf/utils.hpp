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


namespace openpifpaf {

inline bool quiet = false;
void set_quiet(bool v=true);
#define OPENPIFPAF_WARN(...) if (!quiet) { TORCH_WARN(__VA_ARGS__); }

}  // namespace openpifpaf
