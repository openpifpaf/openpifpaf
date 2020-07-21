#include "openpifpaf_postprocessor.hpp"
#include "sample_data_0_8.hpp"

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace lpdnn;

static void app_display_objects(const ai_app::Object_detection::Result& result, float threshold) {

  // Check if nothing found
  int numBoxes = result.items.size();
  if (numBoxes == 1 && result.items[0].confidence < 0) numBoxes = 0;

  cout << "Object detection: " << numBoxes << " objects found." << endl;

  // Add bounding boxes and labels on image
  for (size_t i = 0; i < numBoxes; i++) {
    const auto& item = result.items[i];
    // Skip detection if confidence is too low
    if (item.confidence < threshold) continue;

    const auto& bb = item.bounding_box;
    auto ix = item.class_index;
    string label_text;
    stringstream stream;
    stream << fixed << setprecision(2) << item.confidence;
    string conf_text = stream.str();
    cout << "r: x=" << bb.origin.x << ",y=" << bb.origin.y << " dx=" << bb.size.x << ",dy=" << bb.size.y << " ("
         << conf_text << ")";
    if (!item.landmarks.points.empty()) {
      cout << " landmarks ";
      for (const auto lm : item.landmarks.points) {
        cout << " [" << lm.position.x << "," << lm.position.y;
        if (lm.confidence >= 0)
          cout << " (" << fixed << setprecision(2) << lm.confidence << ")";
        cout << "]";
      }
    }
    cout << endl;

  }
}

int main(int argc, char** argv) {
  lpdnn::aiapp_impl::OpenPifPafPostprocessor pp;
  const ai_app::Rect tileCoordinates { {0, -27}, {640, 481} };
  ai_app::Object_detection::Result res = pp.postprocess_0_8(
        inputWidth, inputHeight, tensorWidth, tensorHeight, tileCoordinates,
        pif_c, pif_r, pif_b, pif_s, paf_c, paf_r1, paf_r2, paf_b1, paf_b2
  );

  cout << "Expected result: " <<
          "2 objects found.\n"
          "r: x=296,y=146 dx=195,dy=223 (0.48) landmarks  [0,-27 (0.00)] [0,-27 (0.00)] [0,-27 (0.00)] [388,150 (0.52)] [405,146 (0.55)] [372,178 (0.82)] [423,178 (0.85)] [338,187 (0.72)] [461,196 (0.90)] [296,191 (0.72)] [491,181 (0.86)] [387,256 (0.74)] [421,257 (0.85)] [383,324 (0.65)] [411,315 (0.70)] [386,369 (0.59)] [403,364 (0.58)]\n"
          "r: x=63,y=303 dx=78,dy=64 (0.27) landmarks  [87,315 (0.42)] [92,309 (0.43)] [83,311 (0.33)] [102,303 (0.53)] [0,-27 (0.00)] [124,315 (0.76)] [73,320 (0.61)] [141,353 (0.68)] [63,352 (0.48)] [117,366 (0.49)] [75,364 (0.37)] [131,363 (0.67)] [94,366 (0.58)] [130,365 (0.31)] [71,367 (0.26)] [0,-27 (0.00)] [0,-27 (0.00)]\n" << endl;

  app_display_objects(res, 0.1);

  return 0;
}
