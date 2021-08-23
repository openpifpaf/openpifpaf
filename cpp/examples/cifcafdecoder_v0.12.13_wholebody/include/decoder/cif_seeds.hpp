#ifndef CIF_SEEDS_HPP
#define CIF_SEEDS_HPP

#include <iostream>
#include "field_config.hpp"

class CifSeeds {
    public:
        vector<Seed> seeds;
        blobnd<float> cifhr;

        float threshold = 0.5;
        float score_scale = 1.0;
        unsigned cif_d1 = 133;
        unsigned cif_d2 = 5;
        unsigned H;
        unsigned W;
    public:
        CifSeeds();
        CifSeeds(blobnd<float> &cifhr, unsigned H, unsigned W);

        void fill(const Fields &fields);
        void fill_cif(float *cif, int stride, float min_scale, const vector<int> &seed_mask);

        vector<Seed> get();

};

#endif