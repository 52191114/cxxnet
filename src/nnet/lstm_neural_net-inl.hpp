#ifndef LSTM_NEURAL_NET_INL_HPP_
#define LSTM_NEURAL_NET_INL_HPP_
/*!
 * \file lstm-neural_net-inl.hpp
 * \brief implementation of common neuralnet
 * \author Bing Xu
 */
#include <vector>
#include <utility>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "../layer/layer.h"
#include "../layer/visitor.h"
#include "../updater/updater.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/thread.h"
#include "./nnet_config.h"
#include "./neural_net-inl.hpp"

namespace cxxnet {
namespace nnet {
/*! \brief implementation of abstract neural net */
template<typename xpu>
struct LSTMNeuralNet : public NeuralNet<xpu> {
  typedef NeuralNet<xpu> Parent;
  LSTMNeuralNet(const NetConfig &cfg,
                mshadow::index_t batch_size,
                int seed,
                mshadow::Stream<xpu> *stream,
                int trunk_size) :
    Parent::cfg(cfg), Parent::max_batch_size(batch_size),
    Parent::rnd(seed), Parent::stream(stream),
    trunk_size(trunk_size) {
    Parent::rnd.set_stream(stream);
    Parent::label_info.name2findex = &cfg.label_name_map;
    snapshots.resize(trunk_size);
    for (size_t i = 0; i < snapshots.size(); ++i) {
      this->SyncChildNetParam(snapshots[i], seed + i);
    }
  }
  ~LSTMNeuralNet() {
    this->FreeSpace();
  }
  inline void InitModel(void) {
    this->InitNet();
    this->ConfigConntions();
    for (size_t i = 0; i < Parent::connections.size(); ++i) {
      if (Parent::cfg.layers[i].name != "") {
        utils::TrackerPrintf("Initializing layer: %s\n", Parent::cfg.layers[i].name.c_str());
      } else {
        utils::TrackerPrintf("Initializing layer: %d\n", static_cast<int>(i));
      }
      layer::Connection<xpu> &c = Parent::connections[i];
      c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
      c.SetStream(Parent::stream);
    }
    for (size_t i = 0; i < Parent::connections.size(); ++i) {
      if (Parent::connections[i].type != layer::kSharedLayer) {
        Parent::connections[i].layer->InitModel();
      }
    }
  }
  inline void LoadModel(utils::IStream &fi) {
    this->FreeSpace();
    this->InitNet();
    this->ConfigConntions();
    for (size_t i = 0; i < Parent::connections.size(); ++i) {
      if (Parent::connections[i].type != layer::kSharedLayer) {
        Parent::connections[i].SetStream(Parent::stream);
        Parent::connections[i].layer->LoadModel(fi);
      }
    }
    for (size_t i = 0; i < Parent::connections.size(); ++i) {
      layer::Connection<xpu> &c = Parent::connections[i];
      c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
      c.SetStream(Parent::stream);
    }
  }
  inline void Forward(bool is_train,
                      mshadow::Tensor<cpu,4> batch,
                      std::vector<mshadow::Tensor<cpu,4> > extra_data,
                      bool need_sync, int t, bool first_trunk) {
    this->PrepForward(t, first_trunk);
    snapshots[t].Forward(is_train, batch, extra_data, need_sync);
    if (t != trunk_size - 1) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        layer::Connection<xpu> &c = Parent::connections[i];
        layer::Connection<xpu> &last_c = snapshots.back().connections[i];
        if (c.type == layer::kLSTM) {
          mshadow::Copy(c.nodes_out[i][0]->data, last_c.nodes_out[i][0]->data, c.nodes_out[i][0]->data.stream_);
          mshadow::Copy(c.nodes_out[i][1]->data, last_c.nodes_out[i][1]->data, c.nodes_out[i][1]->data.stream_);
        }
      }
    }
  }
  inline void Backprop(bool prop_to_input,
                       bool need_update,
                       long update_epoch, int t) {
    this->PrepBackprop(t);
    snapshots[t].Backprop(prop_to_input, need_update, update_epoch, t);
  }
  inline void InitNodes(void) {
    Parent::InitNodes();
    for (size_t i = 0; i < extra_nodes.size(); ++i) {
      extra_nodes[i].AllocSpace();
    }
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i].InitNodes();
    }
  }
  inline void SyncChildNetParam(LSTMNeuralNet &net, int seed) {
    net.cfg = Parent::cfg;
    net.max_batch_size = Parent::batch_size;
    net.rnd.Seed(seed);
    net.stream = Parent::stream;
    net.label_info.name2findex = &Parent::cfg.label_name_map;
    net.trunk_size = 0;
  }
  inline void InitNet(void) {
    Parent::nodes.resize(Parent::cfg.param.num_nodes);
    mshadow::Shape<3> s = Parent::cfg.param.input_shape;
    // setup input shape
    Parent::nodes[0].data.shape_ = mshadow::Shape4(Parent::max_batch, s[0], s[1], s[2]);
    // setup extra data
    for (int i = 0; i < Parent::cfg.param.extra_data_num; ++i) {
      const std::vector<int>& extra_shape = Parent::cfg.extra_shape;
      Parent::nodes[i + 1].data.shape_ = mshadow::Shape4(
        Parent::max_batch, extra_shape[i * 3], extra_shape[i * 3 + 1], extra_shape[i * 3 + 2]);
    }
    // input layer
    for (int i = 0; i < Parent::cfg.param.num_layers; ++i) {
      const NetConfig::LayerInfo &info = Parent::cfg.layers[i];
      layer::Connection<xpu> c;
      c.type = info.type;
      for (size_t j = 0; j < info.nindex_in.size(); ++j) {
        c.nodes_in.push_back(&Parent::nodes[info.nindex_in[j]]);
      }
      for (size_t j = 0; j < info.nindex_out.size(); ++j) {
        c.nodes_out.push_back(&Parent::nodes[info.nindex_out[j]]);
      }
      if (c.type == layer::kLSTM) {
        c.nodes_in.resize(3);
        c.nodes_out.resize(7);
        size_t nd_size = extra_nodes.size();
        extra_nodes.resize(nd_size + 1);
        c.nodes_out[1] = &extra_nodes[nd_size];
        for (size_t  d = 2; d < 7; ++d) {
          if (trunk_size > 0) {
            nd_size = extra_nodes.size();
            extra_nodes.resize(nd_size + 1);
            c.nodes_out[d] = &extra_nodes[nd_size];
          } else {
            c.nodes_out[d] = NULL;
          }
        }
      }
      if (trunk_size == 0) {
        c.layer = NULL;
        continue;
      }
      if (c.type == layer::kSharedLayer) {
        CHECK(info.primary_layer_index >= 0) << "primary_layer_index problem";
        utils::Check(info.primary_layer_index < static_cast<int>(Parent::connections.size()),
                     "shared layer primary_layer_index exceed bound");
        c.layer = Parent::connections[info.primary_layer_index].layer;
        utils::Check(c.layer->AllowSharing(),
                     "some layer you set shared do not allow sharing");
      } else {
        c.layer = layer::CreateLayer(c.type, &Parent::rnd, &Parent::label_info);
      }
      Parent::connections.push_back(c);
    }
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i].InitNet();
    }
  }
  inline void ConfigConntions(void) {
    Parent::ConfigConntions();
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i].ConfigConntions();
    }
  }
  inline void AdjustBatchSize(mshadow::index_t batch_size) {
    if (batch_size != extra_nodes[0].data.size(0)) {
      for (size_t i = 0; i < extra_nodes.size(); ++i) {
        extra_nodes[i].data.shape_[0] = batch_size;
      }
    }
    Parent::AdjustBatchSize(batch_size);
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i].AdjustBatchSize(batch_size);
    }
  }
  inline void FreeSpace() {
    Parent::stream->Wait();
    for (size_t i = 0; i < extra_nodes.size(); ++i) {
      extra_nodes[i].FreeSpace();
    }
    extra_nodes.clear();
    Parent::FreeSpace();
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i].FreeSpace();
    }
  }
private:
  inline void PrepForward(int t, bool first_trunk) {
    if (first_trunk) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        Parent::connections[i].nodes_out[0]->data = 0.0f;
        if (Parent::connections[i].type == layer::kLSTM) {
          Parent::connections[i].nodes_out[1]->data = 0.0f;
        }
      }
    }
    for (size_t i = 0; i < snapshots[t].connections.size(); ++i) {
      layer::Connection<xpu> &now_c  = snapshots[t].connections[i];
      layer::Connection<xpu> &last_c = t == 0 ? Parent::connections[i] : snapshots[t - 1].connections[i];
      now_c.layer = last_c.layer;
      if (now_c.type == layer::kLSTM) {
        now_c.nodes_in[1] = last_c.nodes_out[0];
        now_c.nodes_in[2] = last_c.nodes_out[1];
      }
    }
  }
  inline void PrepBackward(int t) {
    if (t == trunk_size - 1) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        if (Parent::connections[i].type = layer::kLSTM) {
          for (int j = 2; j < 7; ++j) {
            Parent::connections[i].nodes_out[j]->data = 0.0f;
          }
        }
      }
    }
    for (size_t i = 0; i < snapshots[t].connections.size(); ++i) {
      layer::Connection<xpu> &now_c  = snapshots[t].connections[i];
      layer::Connection<xpu> &last_c = t == trunk_size - 1 ? Parent::connections[i] : snapshots[t + 1].connections[i];
      if (now_c.type = layer::kLSTM) {
        for (int j = 2; j < 7; ++j) {
          now_c.nodes_out[j] = last_c.nodes_out[j];
        }
      }
    }
  }
  int trunk_size;
  std::vector<layer::Node<xpu> > extra_nodes;
  std::vector<LSTMNeuralNet> snapshots;
};
} // namespace nnet
} // namespace cxxnet
#endif  // CXXNET_NNET_NEURAL_NET_INL_HPP_
