#ifndef CIF_HR_HPP
#define CIF_HR_HPP

#include "field_config.hpp"


class CifHr {
    public:
        CifMetas meta;
        blobnd<float> accumulated;

        float v_threshold = 0.1;
        int neighbors = 16;

        unsigned H;
        unsigned W;
        unsigned cif_d1 = 133;
        unsigned cif_d2 = 5;
    public:
        CifHr();
        CifHr(unsigned H, unsigned W, const CifMetas &meta);

        void fill_single(const Fields &all_fields);
        void fill(const Fields &all_fields);
        void accumulate(
            int len_cifs,
            blobnd<float> &t, int t_index,
            blobnd<float> &p, int p_index,
            int stride,
            float min_scale
        );
};

#endif