#include "openpifpaf_postprocessor.hpp"

#include <string>
#include <iostream>

using namespace std;
using namespace lpdnn;

int main(int argc, char** argv) {
  lpdnn::aiapp_impl::OpenPifPafPostprocessor pp;
  constexpr int inputWidth = 129;
  constexpr int inputHeight = 97;
  constexpr int tensorWidth = 17;
  constexpr int tensorHeight = 13;
  const ai_app::Rect tileCoordinates { {0, 0}, {inputWidth, inputHeight}};
  constexpr int tensorSize = tensorWidth * tensorHeight;
  static const float pif_c[17 * tensorSize] = {0.0};
  static const float pif_r[17 * 2 * tensorSize] = {0.0};
  static const float pif_b[17 * tensorSize] = {0.0};
  static const float pif_s[17 * tensorSize] = {0.0};
  static const float paf_c[19 * tensorSize] = {0.0};
  static const float paf_r1[19 * 2 * tensorSize] = {0.0};
  static const float paf_r2[19 * 2 * tensorSize] = {0.0};
  static const float paf_b1[19 * tensorSize] = {0.0};
  static const float paf_b2[19 * tensorSize] = {0.0};
  
  pp.postprocess_0_8(inputWidth, inputHeight, tensorWidth, tensorHeight, tileCoordinates,
                     pif_c, pif_r, pif_b, pif_s, paf_c, paf_r1, paf_r2, paf_b1, paf_b2);

  return 0;
}
