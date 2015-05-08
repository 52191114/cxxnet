#ifndef THREAD_NEURAL_NET_INL_HPP_
#define THREAD_NEURAL_NET_INL_HPP_
/*!
 * \file neural_net-inl.hpp
 * \brief implementation of common neuralnet
 * \author Tianqi Chen
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
#include "./lstm_neural_net-inl.hpp"

namespace cxxnet {
namespace nnet {

/*!
 * \brief neural net that runs with an independent thread backed by NeuralNet
 * \tparam
 */
template<typename xpu>
class NeuralNetThread {
 public:
  /*! \brief create a new neural net thread on specific device */
  NeuralNetThread(const NetConfig &cfg,
                  mshadow::ps::ISharedModel<xpu, real_t> *ps,
                  int device_id,
                  mshadow::index_t batch_size,
                  int seed,
                  bool new_thread = true)
      : cfg(cfg), pserver(ps),
        device_id(device_id), batch_size(batch_size),
        seed(seed), new_thread(new_thread) {
    net_ = NULL;
    if (new_thread) {
      destroy_signal = false;
      job_start.Init(0);
      job_end.Init(0);
      worker_thread.Start(ThreadEntry, this);
      // wait until net is created
      job_end.Wait();
    } else {
      mshadow::InitTensorEngine<xpu>(device_id);
      stream = mshadow::NewStream<xpu>();
      net_ = new NeuralNet<xpu>(cfg, batch_size, seed, stream);
    }
  }
  // destructor
  ~NeuralNetThread(void) {
    if (net_ != NULL) {
      if (new_thread) {
        destroy_signal = true;
        job_start.Post();
        worker_thread.Join();
        job_start.Destroy();
        job_end.Destroy();
      } else {
        delete net_;
        mshadow::DeleteStream(stream);
        mshadow::ShutdownTensorEngine<xpu>();
      }
    }
  }

  /*!
   * \brief wait till the the thread finishes current task
   * This function MUST be called every time before running next job
   */
  inline void WaitJob(void) {
    if (new_thread) job_end.Wait();
  }
  inline void InitModel(void) {
    this->task = kInitModel;
    this->ExecTask();
  }
  inline void SaveModel(utils::IStream &fo) {
    iparam_fp = &fo;
    this->task = kSaveModel;
    this->ExecTask();
  }
  inline void LoadModel(utils::IStream &fi) {
    iparam_fp = &fi;
    this->task = kLoadModel;
    this->ExecTask();
  }
  inline void Update(size_t epoch) {
    iparam_epoch = epoch;
    this->task = kUpdate;
    this->ExecTask();
  }
  inline void SyncUpdate(size_t epoch) {
    iparam_epoch = epoch;
    this->task = kUpdate;
    this->ExecTask();
  }
  inline void StartRound(int round) {
    iparam_epoch = static_cast<size_t>(round);
    this->task = kStartRound;
    this->ExecTask();
  }
  inline void SyncParam(void) {
    this->task = kSyncParam;
    this->ExecTask();
  }
  /*! \brief run a training forward backprop pass */
  inline void TrainForwardBackprop(mshadow::Tensor<cpu,4> batch,
                                   const std::vector<mshadow::Tensor<mshadow::cpu, 4> >& extra_data,
                                   const layer::LabelInfo &label_info,
                                   const std::vector<std::pair<int, mshadow::Tensor<cpu, 4> > >& req,
                                   bool prop_to_input,
                                   bool need_sync,
                                   bool need_update,
                                   size_t update_epoch) {
    CHECK(net_ != NULL);
    net_->label_info = label_info;
    iparam_batch = batch;
    iparam_flag = prop_to_input;
    oparam_req = req;
    iparam_need_sync = need_sync;
    iparam_need_update = need_update;
    iparam_epoch = update_epoch;
    iparam_extra_data = extra_data;
    this->task = kTrainProp;
    this->ExecTask();
  }
  /*! \brief run a predicting forward pass, copy final layer  */
  inline void PredictForward(mshadow::Tensor<cpu, 4> batch,
                             const std::vector<mshadow::Tensor<mshadow::cpu, 4> > &extra_data) {
    iparam_batch = batch;
    iparam_extra_data = extra_data;
    this->task = kPredForward;
    this->ExecTask();
  }
  // copy node data out
  inline void CopyNodeData(int nid, mshadow::Tensor<cpu, 4> out_data) {
    iparam_nid = nid;
    oparam_node = out_data;
    this->task = kCopyNode;
    this->ExecTask();
  }
  // copy layer from a fs
  inline void CopyLayer(int lid, utils::IStream &fi) {
    iparam_fp = &fi;
    iparam_lid = lid;
    this->task = kCopyLayer;
    this->ExecTask();
  }
  // set weight into certain layer
  inline void SetWeight(int lid,
                        mshadow::Tensor<cpu, 2> weight,
                        const char *tag) {
    iparam_lid = lid;
    iparam_weight = weight;
    iparam_tag = tag;
    this->task = kSetWeight;
    this->ExecTask();
  }

  // set weight into certain layer
  inline void GetWeight(int lid,
                        mshadow::TensorContainer<cpu, 2> *out_weight,
                        std::vector<index_t> *out_shape,
                        const char *tag) {
    iparam_lid = lid;
    oparam_weight = out_weight;
    oparam_shape = out_shape;
    iparam_tag = tag;
    this->task = kGetWeight;
    this->ExecTask();
  }
  // return reference of node
  inline const NeuralNet<xpu> &net(void) const{
    return *net_;
  }

 private:
  // type of task that can be executed
  enum TaskType {
    kInitModel,
    kLoadModel,
    kSaveModel,
    kUpdate,
    kStartRound,
    kTrainProp,
    kPredForward,
    kCopyNode,
    kCopyLayer,
    kSetWeight,
    kGetWeight,
    kSyncParam
  };
  // thread related code
  inline static CXXNET_THREAD_PREFIX ThreadEntry(void *pthread) {
    static_cast<NeuralNetThread<xpu>*>(pthread)->RunThread();
    utils::ThreadExit(NULL);
    return NULL;
  }
  inline void RunThread(void) {
    mshadow::InitTensorEngine<xpu>(device_id);
    stream = mshadow::NewStream<xpu>();
    // allocate net
    net_ = new NeuralNet<xpu>(cfg, batch_size, seed, stream);
    // tell the master that net is created
    job_end.Post();
    while (!destroy_signal) {
      job_start.Wait();
      if (destroy_signal) break;
      this->TaskDispatch();
      job_end.Post();
    }
    delete net_;
    mshadow::DeleteStream(stream);
    mshadow::ShutdownTensorEngine<xpu>();
  }
  inline void ExecTask(void) {
    if (new_thread) {
      job_start.Post();
    } else {
      this->TaskDispatch();
    }
  }
  inline void TaskDispatch(void) {
    CHECK(net_ != NULL);
    switch (task) {
      case kInitModel: {
        net_->InitModel();
        net_->InitUpdaters(pserver, device_id);
        net_->InitNodes();
        stream->Wait();
        return;
      }
      case kLoadModel: {
        net_->LoadModel(*iparam_fp);
        net_->InitUpdaters(pserver, device_id);
        net_->InitNodes();
        stream->Wait();
        return;
      }
      case kSaveModel: net_->SaveModel(*iparam_fp); return;
      case kUpdate: net_->Update(iparam_epoch); return;
      case kStartRound: net_->StartRound(static_cast<int>(iparam_epoch)); return;
      case kSyncParam: net_->SyncParam(); return;
      case kTrainProp: {
        if (iparam_batch.size(0) == 0) return;
        net_->Forward(true, iparam_batch, iparam_extra_data, iparam_need_sync);
        for (index_t i = 0; i < oparam_req.size(); ++i) {
          index_t id = oparam_req[i].first + (oparam_req[i].first < 0 ? net_->nodes.size() : 0);
          CHECK(id < net_->nodes.size());
          mshadow::Copy(oparam_req[i].second, net_->nodes[id].data, stream);
        }
        net_->Backprop(iparam_flag, iparam_need_update, iparam_epoch);
        stream->Wait();
        return;
      }
      case kPredForward: {
        net_->Forward(false, iparam_batch, iparam_extra_data, true);
        return;
      }
      case kCopyNode: {
        if (iparam_nid < 0) iparam_nid += static_cast<int>(net_->nodes.size());
        CHECK(iparam_nid < static_cast<int>(net_->nodes.size()));
        mshadow::Copy(oparam_node, net_->nodes[iparam_nid].data, stream);
        stream->Wait();
        return;
      }
      case kCopyLayer: {
        CHECK(iparam_lid < static_cast<int>(net_->connections.size()));
        net_->connections[iparam_lid].layer->LoadModel(*iparam_fp);
        return;
      }
      case kSetWeight: {
        CHECK(iparam_lid < static_cast<int>(net_->connections.size()));
        mshadow::TensorContainer<xpu, 2> tmp;
        tmp.Resize(iparam_weight.shape_);
        mshadow::Copy(tmp, iparam_weight, stream);
        stream->Wait();
        std::vector<mshadow::Tensor<xpu, 2> > data;
        data.push_back(tmp);
        layer::SetWeightVisitor<xpu> vs(data, "weight", iparam_tag.c_str());
        net_->connections[iparam_lid].layer->ApplyVisitor(&vs);
        return;
      }
      case kGetWeight: {
        CHECK(iparam_lid < static_cast<int>(net_->connections.size()));
        layer::GetWeightVisitor<xpu> vs("weight", iparam_tag.c_str());
        net_->connections[iparam_lid].layer->ApplyVisitor(&vs);
        if (vs.data.size() == 0) {
          oparam_shape->resize(0);
          oparam_weight->Resize(mshadow::Shape2(0, 0));
        } else {
          oparam_weight->Resize(vs.data[0].shape_);
          mshadow::Copy(*oparam_weight, vs.data[0], stream);
          *oparam_shape = vs.shapes[0];
          CHECK(vs.fields[0] == iparam_tag)
              << " GetWeight:shape mismatch";
          stream->Wait();
        }
        return;
      }
    }
  }
  // the following are fields that are used to pass parameters in or out
  // used to copy out fields in the last layer
  mshadow::Tensor<cpu, 4> oparam_node;
  // used to copy out fields in a given layer
  std::vector<std::pair<int, mshadow::Tensor<cpu, 4> > > oparam_req;
  // output weight parameter
  mshadow::TensorContainer<cpu, 2> *oparam_weight;
  // output shape parameter
  std::vector<index_t> *oparam_shape;
  // input flag
  bool iparam_flag;
  // special input flag for update
  bool iparam_need_sync, iparam_need_update;
  // input epochs
  size_t iparam_epoch;
  // input node id
  int iparam_nid;
  // input layer id
  int iparam_lid;
  // input parameters of file pointers
  utils::IStream *iparam_fp;
  // input batch
  mshadow::Tensor<cpu, 2> iparam_weight;
  // input tag
  std::string iparam_tag;
  // input batch
  mshadow::Tensor<cpu, 4> iparam_batch;
  // input extra data
  std::vector<mshadow::Tensor<cpu,4> > iparam_extra_data;
  // current task
  TaskType task;
  // intenal net implementation
  NeuralNet<xpu> *net_;
  // configuration
  const NetConfig &cfg;
  // signal the destruction of object
  bool destroy_signal;
  // signal of jobs
  utils::Semaphore job_end, job_start;
  // thread object
  utils::Thread worker_thread;
  // parameter server
  mshadow::ps::ISharedModel<xpu, real_t> *pserver;
  // stream used for computation
  mshadow::Stream<xpu> *stream;
  // device id used to intialize tensor engine
  int device_id;
  // local batch size of this thread
  mshadow::index_t batch_size;
  // seed used to intialize this thread
  int seed;
  // whether the implementation is backed by a new thread
  const bool new_thread;
};
}  // namespace nnet
}  // namespace cxxnet
#endif  // CXXNET_NNET_NEURAL_NET_INL_HPP_
