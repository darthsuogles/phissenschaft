#ifndef CUSTOM_CLIP_PLUGIN_H
#define CUSTOM_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>


using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginExt and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class ClipPlugin : public IPluginExt
{
public:
    ClipPlugin(const std::string name, float clipMin, float clipMax);

    ClipPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make ClipPlugin without arguments, so we delete default constructor.
    ClipPlugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginExt* clone() const override;

private:
    const std::string mLayerName;
    float mClipMin, mClipMax;
    size_t mInputVolume;
};

class ClipPluginCreator : public IPluginCreator
{
public:
    ClipPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

#endif
