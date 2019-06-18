#include "pluginImplement.h"
#include "mathFunctions.h"
#include <vector>
#include <algorithm>

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "Permute_1"))
    {
        assert(Permute_1_layer.get() == nullptr);
        Permute_1_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_1_layer.get();
    }
    else if (!strcmp(layerName, "Permute_2"))
    {
        assert(Permute_2_layer.get() == nullptr);
        Permute_2_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_2_layer.get();
    }
    else if (!strcmp(layerName, "Permute_3"))
    {
        assert(Permute_3_layer.get() == nullptr);
        Permute_3_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_3_layer.get();
    }
    else if (!strcmp(layerName, "Permute_4"))
    {
        assert(Permute_4_layer.get() == nullptr);
        Permute_4_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_4_layer.get();
    }
    else if (!strcmp(layerName, "Permute_5"))
    {
        assert(Permute_5_layer.get() == nullptr);
        Permute_5_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_5_layer.get();
    }
    else if (!strcmp(layerName, "Permute_6"))
    {
        assert(Permute_6_layer.get() == nullptr);
        Permute_6_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_6_layer.get();
    }
    else if (!strcmp(layerName, "Permute_7"))
    {
        assert(Permute_7_layer.get() == nullptr);
        Permute_7_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_7_layer.get();
    }
    else if (!strcmp(layerName, "Permute_8"))
    {
        assert(Permute_8_layer.get() == nullptr);
        Permute_8_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_8_layer.get();
    }
    else if (!strcmp(layerName, "Permute_9"))
    {
        assert(Permute_9_layer.get() == nullptr);
        Permute_9_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_9_layer.get();
    }
    else if (!strcmp(layerName, "Permute_10"))
    {
        assert(Permute_10_layer.get() == nullptr);
        Permute_10_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_10_layer.get();
    }
    else if (!strcmp(layerName, "Permute_11"))
    {
        assert(Permute_11_layer.get() == nullptr);
        Permute_11_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_11_layer.get();
    }
    else if (!strcmp(layerName, "Permute_12"))
    {
        assert(Permute_12_layer.get() == nullptr);
        Permute_12_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return Permute_12_layer.get();
    }

//flatten
    else if (!strcmp(layerName, "View_1"))
    {
        assert(View_1_layer.get() == nullptr);
        View_1_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_1_layer.get();
    }
    else if (!strcmp(layerName, "View_2"))
    {
        assert(View_2_layer.get() == nullptr);
        View_2_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_2_layer.get();
    }
    else if (!strcmp(layerName, "View_3"))
    {
        assert(View_3_layer.get() == nullptr);
        View_3_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_3_layer.get();
    }
    else if (!strcmp(layerName, "View_4"))
    {
        assert(View_4_layer.get() == nullptr);
        View_4_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_4_layer.get();
    }
    else if (!strcmp(layerName, "View_5"))
    {
        assert(View_5_layer.get() == nullptr);
        View_5_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_5_layer.get();
    }
    else if (!strcmp(layerName, "View_6"))
    {
        assert(View_6_layer.get() == nullptr);
        View_6_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_6_layer.get();
    }
    else if (!strcmp(layerName, "View_7"))
    {
        assert(View_7_layer.get() == nullptr);
        View_7_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_7_layer.get();
    }
    else if (!strcmp(layerName, "View_8"))
    {
        assert(View_8_layer.get() == nullptr);
        View_8_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_8_layer.get();
    }
    else if (!strcmp(layerName, "View_9"))
    {
        assert(View_9_layer.get() == nullptr);
        View_9_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_9_layer.get();
    }
    else if (!strcmp(layerName, "View_10"))
    {
        assert(View_10_layer.get() == nullptr);
        View_10_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_10_layer.get();
    }
    else if (!strcmp(layerName, "View_11"))
    {
        assert(View_11_layer.get() == nullptr);
        View_11_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_11_layer.get();
    }
    else if (!strcmp(layerName, "View_12"))
    {
        assert(View_12_layer.get() == nullptr);
        View_12_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_12_layer.get();
    }
    else if (!strcmp(layerName, "View_13"))
    {
        assert(View_13_layer.get() == nullptr);
        View_13_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_13_layer.get();
    }
    else if (!strcmp(layerName, "View_14"))
    {
        assert(View_14_layer.get() == nullptr);
        View_14_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return View_14_layer.get();
    }
    else if (!strcmp(layerName, "reshape_cls"))
    {
        assert(reshape_cls.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        reshape_cls = std::unique_ptr<Reshape<5>>(new Reshape<5>());
        return reshape_cls.get();
    }
    else if (!strcmp(layerName, "reshape_pre"))
    {
        assert(reshape_pre.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        reshape_pre = std::unique_ptr<Reshape<4>>(new Reshape<4>());
        return reshape_pre.get();
    }
//softmax

    else if (!strcmp(layerName, "out"))
    {
        assert(mPluginSoftmax == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mPluginSoftmax.get();
    }
    else
    {
        std::cout << "not found  " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "Permute_1"))
    {
        assert(Permute_1_layer.get() == nullptr);
        Permute_1_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_1_layer.get();
    }
    else if (!strcmp(layerName, "Permute_2"))
    {
        assert(Permute_2_layer.get() == nullptr);
        Permute_2_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_2_layer.get();
    }
    //ssd_pruning
    else if (!strcmp(layerName, "Permute_3"))
    {
        assert(Permute_3_layer.get() == nullptr);
        Permute_3_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_3_layer.get();
    }
    else if (!strcmp(layerName, "Permute_4"))
    {
        assert(Permute_4_layer.get() == nullptr);
        Permute_4_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_4_layer.get();
    }
    else if (!strcmp(layerName, "Permute_5"))
    {
        assert(Permute_5_layer.get() == nullptr);
        Permute_5_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_5_layer.get();
    }
    else if (!strcmp(layerName, "Permute_6"))
    {
        assert(Permute_6_layer.get() == nullptr);
        Permute_6_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_6_layer.get();
    }
    else if (!strcmp(layerName, "Permute_7"))
    {
        assert(Permute_7_layer.get() == nullptr);
        Permute_7_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_7_layer.get();
    }
    else if (!strcmp(layerName, "Permute_8"))
    {
        assert(Permute_8_layer.get() == nullptr);
        Permute_8_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_8_layer.get();
    }
    else if (!strcmp(layerName, "Permute_9"))
    {
        assert(Permute_9_layer.get() == nullptr);
        Permute_9_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_9_layer.get();
    }
    else if (!strcmp(layerName, "Permute_10"))
    {
        assert(Permute_10_layer.get() == nullptr);
        Permute_10_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_10_layer.get();
    }
    else if (!strcmp(layerName, "Permute_11"))
    {
        assert(Permute_11_layer.get() == nullptr);
        Permute_11_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_11_layer.get();
    }
    else if (!strcmp(layerName, "Permute_12"))
    {
        assert(Permute_12_layer.get() == nullptr);
        Permute_12_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return Permute_12_layer.get();
    }
//flatten
    else if (!strcmp(layerName, "View_1"))
    {
        assert(View_1_layer.get() == nullptr);
        View_1_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_1_layer.get();
    }
    else if (!strcmp(layerName, "View_2"))
    {
        assert(View_2_layer.get() == nullptr);
        View_2_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_2_layer.get();
    }
    else if (!strcmp(layerName, "View_3"))
    {
        assert(View_3_layer.get() == nullptr);
        View_3_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_3_layer.get();
    }
    else if (!strcmp(layerName, "View_4"))
    {
        assert(View_4_layer.get() == nullptr);
        View_4_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_4_layer.get();
    }
    else if (!strcmp(layerName, "View_5"))
    {
        assert(View_5_layer.get() == nullptr);
        View_5_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_5_layer.get();
    }
    else if (!strcmp(layerName, "View_6"))
    {
        assert(View_6_layer.get() == nullptr);
        View_6_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_6_layer.get();
    }
    else if (!strcmp(layerName, "View_7"))
    {
        assert(View_7_layer.get() == nullptr);
        View_7_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_7_layer.get();
    }
    else if (!strcmp(layerName, "View_8"))
    {
        assert(View_8_layer.get() == nullptr);
        View_8_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_8_layer.get();
    }
    else if (!strcmp(layerName, "View_9"))
    {
        assert(View_9_layer.get() == nullptr);
        View_9_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_9_layer.get();
    }
    else if (!strcmp(layerName, "View_10"))
    {
        assert(View_10_layer.get() == nullptr);
        View_10_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_10_layer.get();
    }
    else if (!strcmp(layerName, "View_11"))
    {
        assert(View_11_layer.get() == nullptr);
        View_11_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_11_layer.get();
    }
    else if (!strcmp(layerName, "View_12"))
    {
        assert(View_12_layer.get() == nullptr);
        View_12_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_12_layer.get();
    }
    else if (!strcmp(layerName, "View_13"))
    {
        assert(View_13_layer.get() == nullptr);
       View_13_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_13_layer.get();
    }
    else if (!strcmp(layerName, "View_14"))
    {
        assert(View_14_layer.get() == nullptr);
       View_14_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return View_14_layer.get();
    }
//reshape
    else if (!strcmp(layerName, "reshape_cls"))
    {
        assert(reshape_cls == nullptr);
        //num of class,by lcg
        reshape_cls = std::unique_ptr<Reshape<5>>(new Reshape<5>(serialData, serialLength));
        return reshape_cls.get();
    }
    else if (!strcmp(layerName, "reshape_pre"))
    {
        assert(reshape_pre == nullptr);
        //num of class,by lcg
        reshape_pre = std::unique_ptr<Reshape<4>>(new Reshape<4>(serialData, serialLength));
        return reshape_pre.get();
    }
//softmax
    else if (!strcmp(layerName, "out"))
    {
        std::cout << "out" << std::endl;
        assert(mPluginSoftmax == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mPluginSoftmax.get();
    }
    else
    {
        std::cout << "else" << std::endl;
        assert(0);
        return nullptr;
    }
}


bool PluginFactory::isPlugin(const char* name)
{
    return (!strcmp(name, "Permute_1")
            || !strcmp(name, "Permute_2")
            || !strcmp(name, "Permute_3")
            || !strcmp(name, "Permute_4")
            || !strcmp(name, "Permute_5")
            || !strcmp(name, "Permute_6")
            || !strcmp(name, "Permute_7")
            || !strcmp(name, "Permute_8")
            || !strcmp(name, "Permute_9")
            || !strcmp(name, "Permute_10")
            || !strcmp(name, "Permute_11")
            || !strcmp(name, "Permute_12")
            || !strcmp(name, "View_1")
            || !strcmp(name, "View_2")
            || !strcmp(name, "View_3")
            || !strcmp(name, "View_4")
            || !strcmp(name, "View_5")
            || !strcmp(name, "View_6")
            || !strcmp(name, "View_7")
            || !strcmp(name, "View_8")
            || !strcmp(name, "View_9")
            || !strcmp(name, "View_10")
            || !strcmp(name, "View_11")
            || !strcmp(name, "View_12")
            || !strcmp(name, "View_13")
            || !strcmp(name, "View_14")
            || !strcmp(name, "reshape_cls")
            || !strcmp(name, "reshape_pre")
            || !strcmp(name, "out")
            );
}



void PluginFactory::destroyPlugin()
{
    std::cout << "distroyPlugin" << std::endl;
    //mNormalizeLayer.release();
    //mNormalizeLayer = nullptr;

    Permute_1_layer.release();
    Permute_1_layer = nullptr;
    Permute_2_layer.release();
    Permute_2_layer = nullptr;
    Permute_3_layer.release();
    Permute_3_layer = nullptr;
    Permute_4_layer.release();
    Permute_4_layer = nullptr;
    Permute_5_layer.release();
    Permute_5_layer = nullptr;
    Permute_6_layer.release();
    Permute_6_layer = nullptr;
    Permute_7_layer.release();
    Permute_7_layer = nullptr;
    Permute_8_layer.release();
    Permute_8_layer = nullptr;
    Permute_9_layer.release();
    Permute_9_layer = nullptr;
    Permute_10_layer.release();
    Permute_10_layer = nullptr;
    Permute_11_layer.release();
    Permute_11_layer = nullptr;
    Permute_12_layer.release();
    Permute_12_layer = nullptr;


    View_1_layer.release();
    View_1_layer = nullptr;
    View_2_layer.release();
    View_2_layer = nullptr;
    View_3_layer.release();
    View_3_layer = nullptr;
    View_4_layer.release();
    View_4_layer = nullptr;
    View_5_layer.release();
    View_5_layer = nullptr;
    View_6_layer.release();
    View_6_layer = nullptr;
    View_7_layer.release();
    View_7_layer = nullptr;
    View_8_layer.release();
    View_8_layer = nullptr;
    View_9_layer.release();
    View_9_layer = nullptr;
    View_10_layer.release();
    View_10_layer = nullptr;
    View_11_layer.release();
    View_11_layer = nullptr;
    View_12_layer.release();
    View_12_layer = nullptr;
    View_13_layer.release();
    View_13_layer = nullptr;
    View_14_layer.release();
    View_14_layer = nullptr;

    reshape_cls.release();
    reshape_cls = nullptr;

    reshape_pre.release();
    reshape_pre = nullptr;

    mPluginSoftmax.release();
    mPluginSoftmax = nullptr;

}
