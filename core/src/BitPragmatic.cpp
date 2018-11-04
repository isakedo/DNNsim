
#include <core/BitPragmatic.h>

namespace core {

    template <typename T>
    static inline
    void computePragmaticProcessingEngine(int n, int m, int x, int y, int i, int j, int k, int stride, int start_batch,
                                          const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_k) {
        for(int channels = 0; channels < std::min(16,max_k); channels++) { // Process 16 synapses in a filter and window
            const T &activation = padded_act.get(n, k + channels, stride * x + i, stride * y + j);
            const T &wheight = wgt.get(m, k + channels - start_batch, i, j);
            // Calculate cycles
        }
    }

    template <typename T>
    static inline
    void computePragmaticTile(int n, int m, std::vector<int> &list_x, std::vector<int> &list_y, int i, int j, int k,
            int stride, int start_batch, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_k) {

        for(int window = 0; window < list_x.size(); window++)   // Process 16 windows
            for(int filter = 0; filter < 16; filter++)  // Process 16 filters
            computePragmaticProcessingEngine<T>(n,m + filter,list_x[window],list_y[window],i,j,k,stride,
                    start_batch,padded_act,wgt,max_k); // 256 computations
    }

    static inline
    bool getWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y) {
        static int x = 0;
        static int y = 0;
        const int max_windows = 16;
        int current_windows = 0;
        for(x; x < out_x; x++) {
            for(y; y < out_y; y++) {
                list_x.push_back(x);
                list_y.push_back(y);
                current_windows++;
                if(current_windows >= max_windows)
                    return true;
            }
        }
        if(current_windows > 0)
            return true;
        // If finish reset values
        x = 0;
        y = 0;
        return false;
    }

    template <typename T>
    void BitPragmatic<T>::computeConvolution(const core::Layer<T> &layer) {
        // Simplify names getting their pointers
        const cnpy::Array<T> &wgt = layer.getWeights();
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array<T> &act = layer.getActivations();
        const std::vector<size_t> &act_shape = act.getShape();

        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
        long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;

        // Set filter batching
        int batches = (int)act.getShape()[1] / (int)wgt_shape[1];
        int it_per_batch = (int)wgt_shape[0] / batches;
        int current_batch = 0, batch_m =0, start_batch = 0;

        // Convolution
        for(int n=0; n<act_shape[0]; n++) {
            for(int m=0; m<wgt_shape[0]; m += 16) { // Sixteen filters each time
                std::vector<int> list_x;
                std::vector<int> list_y;
                while(getWindows(out_x,out_y,list_x,list_y)) { // Sixteen activations each time
                    // Compute in parallel
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            // Sixteen values depthwise
                            for (int k = start_batch; k < wgt_shape[1] + start_batch; k += 16) {
                                computePragmaticTile<T>(n, m, list_x, list_y, i, j, k, stride, start_batch,
                                        padded_act, wgt, (int)act.getShape()[1]);
                            }
                        }
                    }
                }
                batch_m++;
                if(batch_m >= it_per_batch) {
                    batch_m = 0;
                    current_batch++;
                    start_batch = (int)wgt_shape[1]*current_batch;
                }
            }
        }


    }

    template <typename T>
    void BitPragmatic<T>::run(const Network<T> &network) {
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                computeConvolution(layer);
            }
        }
    }

    template class BitPragmatic<uint16_t>;

}