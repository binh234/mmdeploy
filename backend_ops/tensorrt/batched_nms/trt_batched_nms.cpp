// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include "trt_batched_nms.hpp"

#include <cstring>

#include "kernel.h"
#include "trt_batched_nms_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
using namespace nvinfer1;
using nvinfer1::plugin::NMSParameters;

namespace {
static const char* NMS_PLUGIN_VERSION{"1"};
static const char* NMS_PLUGIN_NAME{"TRTBatchedNMS"};
}  // namespace

TRTBatchedNMS::TRTBatchedNMS(const std::string& name, NMSParameters params)
    : TRTPluginBase(name), param(params) {}

TRTBatchedNMS::TRTBatchedNMS(const std::string& name, const void* data,
                             size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &param);
  deserialize_value(&data, &length, &boxesSize);
  deserialize_value(&data, &length, &scoresSize);
  deserialize_value(&data, &length, &numPriors);
  deserialize_value(&data, &length, &mClipBoxes);
}

int TRTBatchedNMS::getNbOutputs() const TRT_NOEXCEPT { return 2; }

nvinfer1::DimsExprs TRTBatchedNMS::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  ASSERT(nbInputs == 2);
  ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
  ASSERT(inputs[0].nbDims == 4);
  ASSERT(inputs[1].nbDims == 3);

  nvinfer1::DimsExprs ret;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.constant(param.keepTopK);
  switch (outputIndex) {
    case 0:
      ret.nbDims = 3;
      ret.d[2] = exprBuilder.constant(5);
      break;
    case 1:
      ret.nbDims = 2;
      break;
    default:
      break;
  }

  return ret;
}

size_t TRTBatchedNMS::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const TRT_NOEXCEPT {
  size_t batch_size = inputs[0].dims.d[0];
  size_t boxes_size =
      inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
  size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
  size_t num_priors = inputs[0].dims.d[1];
  bool shareLocation = (inputs[0].dims.d[2] == 1);
  int topk = param.topK > 0 && param.topK <= inputs[1].dims.d[1]
                 ? topk
                 : inputs[1].dims.d[1];
  return detectionInferenceWorkspaceSize(
      shareLocation, batch_size, boxes_size, score_size, param.numClasses,
      num_priors, topk, DataType::kFLOAT, DataType::kFLOAT);
}

int TRTBatchedNMS::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                           const nvinfer1::PluginTensorDesc* outputDesc,
                           const void* const* inputs, void* const* outputs,
                           void* workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  const void* const locData = inputs[0];
  const void* const confData = inputs[1];

  void* nmsedDets = outputs[0];
  void* nmsedLabels = outputs[1];

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size =
      inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];
  bool shareLocation = (inputDesc[0].dims.d[2] == 1);

  int topk = param.topK > 0 && param.topK <= inputDesc[1].dims.d[1]
                 ? topk
                 : inputDesc[1].dims.d[1];
  pluginStatus_t status = nmsInference(
      stream, batch_size, boxes_size, score_size, shareLocation,
      param.backgroundLabelId, num_priors, param.numClasses, topk,
      param.keepTopK, param.scoreThreshold, param.iouThreshold,
      DataType::kFLOAT, locData, DataType::kFLOAT, confData, nmsedDets,
      nmsedLabels, workSpace, param.isNormalized, false, mClipBoxes);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

size_t TRTBatchedNMS::getSerializationSize() const TRT_NOEXCEPT {
  // NMSParameters, boxesSize,scoresSize,numPriors
  return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void TRTBatchedNMS::serialize(void* buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, param);
  serialize_value(&buffer, boxesSize);
  serialize_value(&buffer, scoresSize);
  serialize_value(&buffer, numPriors);
  serialize_value(&buffer, mClipBoxes);
}

void TRTBatchedNMS::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments
}

bool TRTBatchedNMS::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* ioDesc, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  if (pos == 3) {
    return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
           ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* TRTBatchedNMS::getPluginType() const TRT_NOEXCEPT {
  return NMS_PLUGIN_NAME;
}

const char* TRTBatchedNMS::getPluginVersion() const TRT_NOEXCEPT {
  return NMS_PLUGIN_VERSION;
}

IPluginV2DynamicExt* TRTBatchedNMS::clone() const TRT_NOEXCEPT {
  auto* plugin = new TRTBatchedNMS(mLayerName, param);
  plugin->boxesSize = boxesSize;
  plugin->scoresSize = scoresSize;
  plugin->numPriors = numPriors;
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

nvinfer1::DataType TRTBatchedNMS::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
  ASSERT(index >= 0 && index < this->getNbOutputs());
  if (index == 1) {
    return nvinfer1::DataType::kINT32;
  }
  return inputTypes[0];
}

void TRTBatchedNMS::setClipParam(bool clip) { mClipBoxes = clip; }

TRTBatchedNMSCreator::TRTBatchedNMSCreator() {
  mPluginAttributes.emplace_back(
      PluginField("background_label_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("is_normalized", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TRTBatchedNMSCreator::getPluginName() const TRT_NOEXCEPT {
  return NMS_PLUGIN_NAME;
}

const char* TRTBatchedNMSCreator::getPluginVersion() const TRT_NOEXCEPT {
  return NMS_PLUGIN_VERSION;
}

IPluginV2Ext* TRTBatchedNMSCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {
  const PluginField* fields = fc->fields;
  bool clipBoxes = true;
  nvinfer1::plugin::NMSParameters params{};

  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "background_label_id")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "num_classes")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.numClasses = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "topk")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.topK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "keep_topk")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.keepTopK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.iouThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "is_normalized")) {
      params.isNormalized = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "clip_boxes")) {
      clipBoxes = *(static_cast<const bool*>(fields[i].data));
    }
  }

  TRTBatchedNMS* plugin = new TRTBatchedNMS(name, params);
  plugin->setClipParam(clipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* TRTBatchedNMSCreator::deserializePlugin(
    const char* name, const void* serialData,
    size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call NMS::destroy()
  TRTBatchedNMS* plugin = new TRTBatchedNMS(name, serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBatchedNMSCreator);
}  // namespace mmdeploy
