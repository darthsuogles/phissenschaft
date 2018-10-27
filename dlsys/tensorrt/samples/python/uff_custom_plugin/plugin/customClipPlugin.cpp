#include "customClipPlugin.h"
#include "NvInfer.h"
#include "clipKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Clip plugin specific constants
namespace {
    static const char* CLIP_PLUGIN_VERSION{"001"};
    static const char* CLIP_PLUGIN_NAME{"CustomClipPlugin"};
}

// Static class fields initialization
PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ClipPluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

ClipPlugin::ClipPlugin(const std::string name, float clipMin, float clipMax)
    : mLayerName(name)
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mClipMin = readFromBuffer<float>(d);
    mClipMax = readFromBuffer<float>(d);

    assert(d == (a + length));
}

const char* ClipPlugin::getPluginType() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPlugin::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

int ClipPlugin::getNbOutputs() const
{
    return 1;
}

Dims ClipPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Validate input arguments
    assert(nbInputDims == 1);
    assert(index == 0);

    // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
    return *inputs;
}

int ClipPlugin::initialize()
{
    return 0;
}

int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);

    return status;
}

size_t ClipPlugin::getSerializationSize()
{
    return 2 * sizeof(float);
}

void ClipPlugin::serialize(void* buffer)
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mClipMin);
    writeToBuffer(d, mClipMax);

    assert(d == a + getSerializationSize());
}

void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void ClipPlugin::terminate() {}

void ClipPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginExt* ClipPlugin::clone() const
{
    return new ClipPlugin(mLayerName, mClipMin, mClipMax);
}

ClipPluginCreator::ClipPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ClipPluginCreator::getPluginName() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

const PluginFieldCollection* ClipPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginExt* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float clipMin, clipMax;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "clipMin") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMin = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "clipMax") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }
    return new ClipPlugin(name, clipMin, clipMax);
}

IPluginExt* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new ClipPlugin(name, serialData, serialLength);
}
