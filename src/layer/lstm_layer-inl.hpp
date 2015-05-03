#ifndef LSTM_LAYER_INL_HPP
#define LSTM_LAYER_INL_HPP
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu,
         typename ZOp, typename ZOpGrad,
         typename IOp, typename IOpGrad,
         typename FOp, typename FOpGrad,
         typename OOp, typename OOpGrad,
         typename COp, typename COpGrad>
class LSTMLayer : public ILayer<xpu> {
  public:
    LSTMLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {}
    virtual ~LSTMLayer() {}
    virtual void SetParam(char *name, char *val) {
      param_.SetParam(name, val);
    }
    virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                                const std::vector<Node<xpu>*> &nodes_out,
                                ConnectState<xpu> *p_cstate) {
      // input: x_t, y_{t-1}, c_{t-1}
      // output: y_t, c_t, delta_z, delta_i, delta_o, delta_f, delta_c
      // state: z_no_act, i_no_act, f_no_act, o_no_act, c_t_no_act
      // Or we can save the act to speed up?
      // dmlc::CHECK(nodes_in.size() == 7);
      // dmlc::CHECK(nodes_out.size() == 7);
      const int batch_size = nodes_in[0]->data.shape_[0];
      mshadow::Shape<4> oshape = mshadow::Shape4(batch_size, 1, 1, param_.num_hidden);
      if (param_.num_input_node == 0) {
        param_.num_input_node = static_cast<int>(nodes_in[0]->data.size(3));
      }
      for (int i = 0; i < nodes_out.size(); ++i) {
        nodes_out[i]->shape_ = oshape;
      }
      p_cstate->states.resize(6);
      for (int i = 0; i < p_cstate->states.size(); ++i) {
        p_cstate->states[i].Resize(oshape);
      }
    }
    virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {

    }
    virtual void InitModel() {
      Wi_.Resize(mshadow::Shape2(param_.num_hidden, param_.num_input_node));
      Ui_.Resize(mshadow::Shape2(param_.num_hidden, param_.num_hidden));
      bi_.Resize(mshadow::Shape2(param_.num_hidden));
      Wf_.Resize(Wi_.shape_);
      Uf_.Resize(Ui_.shape_);
      bf_.Resize(bi_.shape_);
      Wo_.Resize(Wi_.shape_);
      Uo_.Resize(Ui_.shape_);
      bo_.Resize(bi_.shape_);
      Wz_.Resize(Wi_.shape_);
      Uz_.Resize(Ui_.shape_);
      bz_.Resize(bi_.shape_);
      gWi_.Resize(Wi_.shape_);
      gUi_.Resize(Ui_.shape_);
      gbi_.Resize(bi_.shape_);
      gWf_.Resize(Wi_.shape_);
      gUf_.Resize(Ui_.shape_);
      gbf_.Resize(bi_.shape_);
      gWo_.Resize(Wi_.shape_);
      gUo_.Resize(Ui_.shape_);
      gbo_.Resize(bi_.shape_);
      gWz_.Resize(Wi_.shape_);
      gUz_.Resize(Ui_.shape_);
      gbz_.Resize(bi_.shape_);
      gWi_ = 0.0f;
      gUi_ = 0.0f;
      gbi_ = 0.0f;
      gWz_ = 0.0f;
      gUz_ = 0.0f;
      gbz_ = 0.0f;
      gWf_ = 0.0f;
      gUf_ = 0.0f;
      gbf_ = 0.0f;
      gWo_ = 0.0f;
      gUo_ = 0.0f;
      gbo_ = 0.0f;
    }
    virtual void SaveModel(utils::IStream &fo) const {
      fo.Write(&param_, sizeof(LSTMParam));
      Wi_.SaveBinary(fo);
      Wf_.SaveBinary(fo);
      Wo_.SaveBinary(fo);
      Wz_.SaveBinary(fo);
      Ui_.SaveBinary(fo);
      Uf_.SaveBinary(fo);
      Uo_.SaveBinary(fo);
      Uz_.SaveBinary(fo);
      bi_.SaveBinary(fo);
      bf_.SaveBinary(fo);
      bo_.SaveBinary(fo);
      bz_.SaveBinary(fo);
    }
    virtal void LoadModel(utils::IStream &fi) {
      utils::Check(fi.Read(&param_, sizeof(LSTMParam)) != 0,
                   "FullConnectLayer:LoadModel invalid model file");
      Wi_.LoadBinary(fi);
      Wf_.LoadBinary(fi);
      Wo_.LoadBinary(fi);
      Wz_.LoadBinary(fi);
      Ui_.LoadBinary(fi);
      Uf_.LoadBinary(fi);
      Uo_.LoadBinary(fi);
      Uz_.LoadBinary(fi);
      bi_.LoadBinary(fi);
      bf_.LoadBinary(fi);
      bo_.LoadBinary(fi);
      bz_.LoadBinary(fi);
      gWi_.Resize(Wi_.shape_);
      gWf_.Resize(Wf_.shape_);
      gWo_.Resize(Wo_.shape_);
      gWz_.Resize(Wc_.shape_);
      gUi_.Resize(Wi_.shape_);
      gUf_.Resize(Wf_.shape_);
      gUo_.Resize(Wo_.shape_);
      gUz_.Resize(Wc_.shape_);
      gbi_.Resize(bi_.shape_);
      gbf_.Resize(bf_.shape_);
      gbo_.Resize(bo_.shape_);
      gbz_.Resize(bc_.shape_);
      gWi_ = 0.0f;
      gUi_ = 0.0f;
      gbi_ = 0.0f;
      gWz_ = 0.0f;
      gUz_ = 0.0f;
      gbz_ = 0.0f;
      gWf_ = 0.0f;
      gUf_ = 0.0f;
      gbf_ = 0.0f;
      gWo_ = 0.0f;
      gUo_ = 0.0f;
      gbo_ = 0.0f;

    }
    virtual void SetStream(mshadow::Stream<xpu> *stream) {
      Wi_.set_stream(stream);
      Wf_.set_stream(stream);
      Wo_.set_stream(stream);
      Wz_.set_stream(stream);
      gWi_.set_stream(stream);
      gWf_.set_stream(stream);
      gWo_.set_stream(stream);
      gWz_.set_stream(stream);
      Ui_.set_stream(stream);
      Uf_.set_stream(stream);
      Uo_.set_stream(stream);
      Uz_.set_stream(stream);
      gUi_.set_stream(stream);
      gUf_.set_stream(stream);
      gUo_.set_stream(stream);
      gUz_.set_stream(stream);
      bi_.set_stream(stream);
      bf_.set_stream(stream);
      bo_.set_stream(stream);
      bz_.set_stream(stream);
      gbi_.set_stream(stream);
      gbf_.set_stream(stream);
      gbo_.set_stream(stream);
      gbz_.set_stream(stream);
    }
    virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
      mshadow::Tensor<xpu, 2> x = nodes_in[0]->mat();
      mshadow::Tensor<xpu, 2> last_c = nodes_in[1]->mat();
      mshadow::Tensor<xpu, 2> last_y = nodes_in[2]->mat();
      mshadow::Tensor<xpu, 2> y = nodes_out[0]->mat();
      mshadow::Tensor<xpu, 2> c_t = nodes_out[1]->mat();
      mshadow::Tensor<xpu, 2> next_y = nodes_out[2]->mat();

      mshadow::Tensor<xpu, 2> z = p_cstate->states[0].FlatTo2D();
      mshadow::Tensor<xpu, 2> i = p_cstate->states[1].FlatTo2D();
      mshadow::Tensor<xpu, 2> f = p_cstate->states[2].FlatTo2D();
      mshadow::Tensor<xpu, 2> o = p_cstate->states[3].FlatTo2D();
      mshadow::Tensor<xpu, 2> c = p_cstate->states[4].FlatTo2D();
      mshadow::Tensor<xpu, 2> y_back = p_cstate->states[5].FlatTo2D();
      index_t nbatch = x.size(0);
      using namespace mshadow::expr;
      z = dot(x, Wz_.T());
      z += dot(last_y, Uz_.T());
      i = dot(x, Wi_.T());
      i += dot(last_y, Ui_.T());
      f = dot(x, Wf_.T());
      f += dot(last_y, Uf_.T());
      o = dot(x, Wo_.T());
      o += dot(last_y, Uo_.T());
      if (param_.no_bias == 0) {
        z += repmat(bz_, nbatch);
        i += repmat(bi_, nbatch);
        f += repmat(bf_, nbatch);
        o += repmat(bo_, nbatch);
      }
      c = F<ZOp>(z) * F<IOp>(i) + last_c * F<FOp>(f);
      c_t = F<COp>(c);
      y = c_t * F<Oop>(o);
      mshadow::Copy(y_back, y, y_back.stream_);
    }

    virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
      using namespace mshadow::expr;
      mshadow::Tensor<xpu, 2> delta_y = nodes_out[0]->data.mat();
      mshadow::Tensor<xpu, 2> next_c = nodes_out[1]->data.mat();
      mshadow::Tensor<xpu, 2> delta_z = nodes_out[3]->data.mat();
      mshadow::Tensor<xpu, 2> delta_i = nodes_out[4]->data.mat();
      mshadow::Tensor<xpu, 2> delta_o = nodes_out[5]->data.mat();
      mshadow::Tensor<xpu, 2> delta_f = nodes_out[6]->data.mat();
      mshadow::Tensor<xpu, 2> delta_c = nodes_out[7]->data.mat();

      mshadow::Tensor<xpu, 2> x = nodes_in[0]->data.mat();

      mshadow::Tensor<xpu, 2> z = p_cstate->states[0].FlatTo2D();
      mshadow::Tensor<xpu, 2> i = p_cstate->states[1].FlatTo2D();
      mshadow::Tensor<xpu, 2> f = p_cstate->states[2].FlatTo2D();
      mshadow::Tensor<xpu, 2> o = p_cstate->states[3].FlatTo2D();
      mshadow::Tensor<xpu, 2> c = p_cstate->states[4].FlatTo2D();
      mshadow::Tensor<xpu, 2> y_back = p_cstate->states[5].FlatTo2D();

      gUi_ += dot(delta_i, y_back);
      gUf_ += dot(delta_f, y_back);
      gUz_ += dot(delta_z, y_back);
      gUo_ += dot(delta_o, y_back);

      delta_y += dot(delta_z, Uz_);
      delta_y += dot(delta_i, Ui_);
      delta_y += dot(delta_f, Uf_);
      delta_y += dot(delta_o, Uo_);

      delta_o = delta_y * F<COp>(c) * F<OOpGrad>(o);
      delta_c = delta_y * F<OOp>(o) * F<COpGrad>(c) + delta_c * delta_f;
      delta_f = delta_c * F<COp>(c) * F<FOpGrad>(f);
      delta_i = delta_c * F<ZOp>(z) * F<IOpGrad>(i);
      delta_z = delta_c * F<IOp>(i) * F<ZOpGrad>(z);

      gWi_ += dot(delta_i.T(), x);
      gWz_ += dot(delta_z.T(), x);
      gWf_ += dot(delta_f.T(), x);
      gWo_ += dot(delta_o.T(), x);

      x = dot(delta_z, Wz_);
      x += dot(delta_i, Wi_);
      x += dot(delta_f, Wf_);
      x += dot(delta_o, Wo_);

      if (param_.no_bias == 0) {
        gbi_ += sum_rows(delta_i);
        gbo_ += sum_rows(delta_o);
        gbf_ += sum_rows(delta_f);
        gbz_ += sum_rows(delta_z);
      }
    }

  protected:
    /*! \brief parameters for LSTM */
    LSTMParam param_;
    /*! \brief random number generator */
    mshadow::Random<xpu> *prnd_;
    /*! \brief weight */
    mshadow::TensorContainer<xpu, 2> Wi_, Wf_, Wo_, Wz_;
    mshadow::TensorContainer<xpu, 2> Ui_, Uf_, Uo_, Uz_;
    /*! \brief bias */
    mshadow::TensorContainer<xpu, 1> bi_, bf_, bo_, bz_;
    /*! \brief weight gradient */
    mshadow::TensorContainer<xpu, 2> gWi_, gWf_, gWo_, gWz_;
    mshadow::TensorContainer<xpu, 2> gUi_, gUf_, gUo_, gUz_;
    /*! \brief bias gradient */
    mshadow::TensorContainer<xpu, 1> gbi_, gbf_, gbo_, gbz_;
}; // class LSTMLayer

} // namespace layer
} // namespace cxxnet
#endif // LSTM_LAYER_INL_HPP
