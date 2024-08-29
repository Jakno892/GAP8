#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "ResizeGenerator.h"

#include "CNN_Copy_Generators.h"





void classificationModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 4, "Gap.h", "classification.h", "CNN_BasicKernels_SQ8.h", "ResizeBasicKernels.h");
    SetGeneratedFilesNames("classificationKernels.c", "classificationKernels.h");


    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "classification_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "classification_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "classification_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "classification_L3_Flash", "classification_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    LoadResizeLibrary();


    CNN_GenControl_T gen_ctrl_S3_Conv2d_1x1x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_1x1x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_1x1x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_1
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_1x1x1x1", &gen_ctrl_S3_Conv2d_1x1x1x1,
                               4, 1,
                               1, 1, 244, 324,
                               KOP_CONV, 1, 1, 1, 1, 2, 2, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S6_Conv2d_3x1x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_3x1x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_3x1x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_2
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_3x1x1x1", &gen_ctrl_S6_Conv2d_3x1x1x1,
                               4, 1,
                               1, 3, 122, 162,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for RESIZE_BILINEAR_0_3
    GenerateResizeMultiChannel("S7_Op_RESIZE_BILINEAR_0_3", 122, 162, 96, 96, 3, SIGNED_INOUT, KOP_BILINEAR_RESIZE);
    
    CNN_GenControl_T gen_ctrl_S10_Conv2d_16x3x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S10_Conv2d_16x3x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S10_Conv2d_16x3x3x3_Relu6, "PADTYPE", AT_OPT_VAL(1));
    // generator for CONV_2D_0_4_fusion
    CNN_ConvolutionPoolAct_SQ8("S10_Conv2d_16x3x3x3_Relu6", &gen_ctrl_S10_Conv2d_16x3x3x3_Relu6,
                               4, 1,
                               3, 16, 96, 96,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_5_fusion
    CNN_ConvolutionPoolAct_SQ8("S13_Conv2d_16x1x3x3_Relu6", 0,
                               4, 1,
                               16, 16, 48, 48,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S16_Conv2d_8x16x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S16_Conv2d_8x16x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S16_Conv2d_8x16x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_6
    CNN_ConvolutionPoolAct_SQ8("S16_Conv2d_8x16x1x1", &gen_ctrl_S16_Conv2d_8x16x1x1,
                               4, 1,
                               16, 8, 48, 48,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S19_Conv2d_48x8x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S19_Conv2d_48x8x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S19_Conv2d_48x8x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_7_fusion
    CNN_ConvolutionPoolAct_SQ8("S19_Conv2d_48x8x1x1_Relu6", &gen_ctrl_S19_Conv2d_48x8x1x1_Relu6,
                               4, 1,
                               8, 48, 48, 48,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S22_Conv2d_48x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S22_Conv2d_48x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S22_Conv2d_48x1x3x3_Relu6, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_9_fusion
    CNN_ConvolutionPoolAct_SQ8("S22_Conv2d_48x1x3x3_Relu6", &gen_ctrl_S22_Conv2d_48x1x3x3_Relu6,
                               4, 1,
                               48, 48, 48, 48,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S25_Conv2d_8x48x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S25_Conv2d_8x48x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S25_Conv2d_8x48x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_10
    CNN_ConvolutionPoolAct_SQ8("S25_Conv2d_8x48x1x1", &gen_ctrl_S25_Conv2d_8x48x1x1,
                               4, 1,
                               48, 8, 24, 24,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S28_Conv2d_48x8x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S28_Conv2d_48x8x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S28_Conv2d_48x8x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_11_fusion
    CNN_ConvolutionPoolAct_SQ8("S28_Conv2d_48x8x1x1_Relu6", &gen_ctrl_S28_Conv2d_48x8x1x1_Relu6,
                               4, 1,
                               8, 48, 24, 24,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_12_fusion
    CNN_ConvolutionPoolAct_SQ8("S31_Conv2d_48x1x3x3_Relu6", 0,
                               4, 1,
                               48, 48, 24, 24,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S34_Conv2d_8x48x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S34_Conv2d_8x48x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S34_Conv2d_8x48x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_13
    CNN_ConvolutionPoolAct_SQ8("S34_Conv2d_8x48x1x1", &gen_ctrl_S34_Conv2d_8x48x1x1,
                               4, 1,
                               48, 8, 24, 24,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_14
    CNN_MatAddAct_SQ8("S35_MatAdd_8x24x24", 0, 8, 24, 24, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S38_Conv2d_48x8x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S38_Conv2d_48x8x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S38_Conv2d_48x8x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_15_fusion
    CNN_ConvolutionPoolAct_SQ8("S38_Conv2d_48x8x1x1_Relu6", &gen_ctrl_S38_Conv2d_48x8x1x1_Relu6,
                               4, 1,
                               8, 48, 24, 24,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S41_Conv2d_48x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S41_Conv2d_48x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S41_Conv2d_48x1x3x3_Relu6, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_17_fusion
    CNN_ConvolutionPoolAct_SQ8("S41_Conv2d_48x1x3x3_Relu6", &gen_ctrl_S41_Conv2d_48x1x3x3_Relu6,
                               4, 1,
                               48, 48, 24, 24,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S44_Conv2d_16x48x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S44_Conv2d_16x48x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S44_Conv2d_16x48x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_18
    CNN_ConvolutionPoolAct_SQ8("S44_Conv2d_16x48x1x1", &gen_ctrl_S44_Conv2d_16x48x1x1,
                               4, 1,
                               48, 16, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S47_Conv2d_96x16x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S47_Conv2d_96x16x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S47_Conv2d_96x16x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_19_fusion
    CNN_ConvolutionPoolAct_SQ8("S47_Conv2d_96x16x1x1_Relu6", &gen_ctrl_S47_Conv2d_96x16x1x1_Relu6,
                               4, 1,
                               16, 96, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_20_fusion
    CNN_ConvolutionPoolAct_SQ8("S50_Conv2d_96x1x3x3_Relu6", 0,
                               4, 1,
                               96, 96, 12, 12,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S53_Conv2d_16x96x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S53_Conv2d_16x96x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S53_Conv2d_16x96x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_21
    CNN_ConvolutionPoolAct_SQ8("S53_Conv2d_16x96x1x1", &gen_ctrl_S53_Conv2d_16x96x1x1,
                               4, 1,
                               96, 16, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_22
    CNN_MatAddAct_SQ8("S54_MatAdd_16x12x12", 0, 16, 12, 12, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S57_Conv2d_96x16x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S57_Conv2d_96x16x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S57_Conv2d_96x16x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_23_fusion
    CNN_ConvolutionPoolAct_SQ8("S57_Conv2d_96x16x1x1_Relu6", &gen_ctrl_S57_Conv2d_96x16x1x1_Relu6,
                               4, 1,
                               16, 96, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_24_fusion
    CNN_ConvolutionPoolAct_SQ8("S60_Conv2d_96x1x3x3_Relu6", 0,
                               4, 1,
                               96, 96, 12, 12,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S63_Conv2d_16x96x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S63_Conv2d_16x96x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S63_Conv2d_16x96x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_25
    CNN_ConvolutionPoolAct_SQ8("S63_Conv2d_16x96x1x1", &gen_ctrl_S63_Conv2d_16x96x1x1,
                               4, 1,
                               96, 16, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_26
    CNN_MatAddAct_SQ8("S64_MatAdd_16x12x12", 0, 16, 12, 12, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S67_Conv2d_96x16x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S67_Conv2d_96x16x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S67_Conv2d_96x16x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_27_fusion
    CNN_ConvolutionPoolAct_SQ8("S67_Conv2d_96x16x1x1_Relu6", &gen_ctrl_S67_Conv2d_96x16x1x1_Relu6,
                               4, 1,
                               16, 96, 12, 12,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S70_Conv2d_96x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S70_Conv2d_96x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S70_Conv2d_96x1x3x3_Relu6, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_29_fusion
    CNN_ConvolutionPoolAct_SQ8("S70_Conv2d_96x1x3x3_Relu6", &gen_ctrl_S70_Conv2d_96x1x3x3_Relu6,
                               4, 1,
                               96, 96, 12, 12,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S73_Conv2d_24x96x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S73_Conv2d_24x96x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S73_Conv2d_24x96x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_30
    CNN_ConvolutionPoolAct_SQ8("S73_Conv2d_24x96x1x1", &gen_ctrl_S73_Conv2d_24x96x1x1,
                               4, 1,
                               96, 24, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S76_Conv2d_144x24x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S76_Conv2d_144x24x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S76_Conv2d_144x24x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_31_fusion
    CNN_ConvolutionPoolAct_SQ8("S76_Conv2d_144x24x1x1_Relu6", &gen_ctrl_S76_Conv2d_144x24x1x1_Relu6,
                               4, 1,
                               24, 144, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_32_fusion
    CNN_ConvolutionPoolAct_SQ8("S79_Conv2d_144x1x3x3_Relu6", 0,
                               4, 1,
                               144, 144, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S82_Conv2d_24x144x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S82_Conv2d_24x144x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S82_Conv2d_24x144x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_33
    CNN_ConvolutionPoolAct_SQ8("S82_Conv2d_24x144x1x1", &gen_ctrl_S82_Conv2d_24x144x1x1,
                               4, 1,
                               144, 24, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_34
    CNN_MatAddAct_SQ8("S83_MatAdd_24x6x6", 0, 24, 6, 6, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S86_Conv2d_144x24x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S86_Conv2d_144x24x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S86_Conv2d_144x24x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_35_fusion
    CNN_ConvolutionPoolAct_SQ8("S86_Conv2d_144x24x1x1_Relu6", &gen_ctrl_S86_Conv2d_144x24x1x1_Relu6,
                               4, 1,
                               24, 144, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_36_fusion
    CNN_ConvolutionPoolAct_SQ8("S89_Conv2d_144x1x3x3_Relu6", 0,
                               4, 1,
                               144, 144, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S92_Conv2d_24x144x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S92_Conv2d_24x144x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S92_Conv2d_24x144x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_37
    CNN_ConvolutionPoolAct_SQ8("S92_Conv2d_24x144x1x1", &gen_ctrl_S92_Conv2d_24x144x1x1,
                               4, 1,
                               144, 24, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_38
    CNN_MatAddAct_SQ8("S93_MatAdd_24x6x6", 0, 24, 6, 6, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S96_Conv2d_144x24x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S96_Conv2d_144x24x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S96_Conv2d_144x24x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_39_fusion
    CNN_ConvolutionPoolAct_SQ8("S96_Conv2d_144x24x1x1_Relu6", &gen_ctrl_S96_Conv2d_144x24x1x1_Relu6,
                               4, 1,
                               24, 144, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_40_fusion
    CNN_ConvolutionPoolAct_SQ8("S99_Conv2d_144x1x3x3_Relu6", 0,
                               4, 1,
                               144, 144, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S102_Conv2d_24x144x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S102_Conv2d_24x144x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S102_Conv2d_24x144x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_41
    CNN_ConvolutionPoolAct_SQ8("S102_Conv2d_24x144x1x1", &gen_ctrl_S102_Conv2d_24x144x1x1,
                               4, 1,
                               144, 24, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_42
    CNN_MatAddAct_SQ8("S103_MatAdd_24x6x6", 0, 24, 6, 6, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S106_Conv2d_144x24x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S106_Conv2d_144x24x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S106_Conv2d_144x24x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_43_fusion
    CNN_ConvolutionPoolAct_SQ8("S106_Conv2d_144x24x1x1_Relu6", &gen_ctrl_S106_Conv2d_144x24x1x1_Relu6,
                               4, 1,
                               24, 144, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_44_fusion
    CNN_ConvolutionPoolAct_SQ8("S109_Conv2d_144x1x3x3_Relu6", 0,
                               4, 1,
                               144, 144, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S112_Conv2d_32x144x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S112_Conv2d_32x144x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S112_Conv2d_32x144x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_45
    CNN_ConvolutionPoolAct_SQ8("S112_Conv2d_32x144x1x1", &gen_ctrl_S112_Conv2d_32x144x1x1,
                               4, 1,
                               144, 32, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S115_Conv2d_192x32x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S115_Conv2d_192x32x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S115_Conv2d_192x32x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_46_fusion
    CNN_ConvolutionPoolAct_SQ8("S115_Conv2d_192x32x1x1_Relu6", &gen_ctrl_S115_Conv2d_192x32x1x1_Relu6,
                               4, 1,
                               32, 192, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_47_fusion
    CNN_ConvolutionPoolAct_SQ8("S118_Conv2d_192x1x3x3_Relu6", 0,
                               4, 1,
                               192, 192, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S121_Conv2d_32x192x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S121_Conv2d_32x192x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S121_Conv2d_32x192x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_48
    CNN_ConvolutionPoolAct_SQ8("S121_Conv2d_32x192x1x1", &gen_ctrl_S121_Conv2d_32x192x1x1,
                               4, 1,
                               192, 32, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_49
    CNN_MatAddAct_SQ8("S122_MatAdd_32x6x6", 0, 32, 6, 6, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S125_Conv2d_192x32x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S125_Conv2d_192x32x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S125_Conv2d_192x32x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_50_fusion
    CNN_ConvolutionPoolAct_SQ8("S125_Conv2d_192x32x1x1_Relu6", &gen_ctrl_S125_Conv2d_192x32x1x1_Relu6,
                               4, 1,
                               32, 192, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_51_fusion
    CNN_ConvolutionPoolAct_SQ8("S128_Conv2d_192x1x3x3_Relu6", 0,
                               4, 1,
                               192, 192, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S131_Conv2d_32x192x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S131_Conv2d_32x192x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S131_Conv2d_32x192x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_52
    CNN_ConvolutionPoolAct_SQ8("S131_Conv2d_32x192x1x1", &gen_ctrl_S131_Conv2d_32x192x1x1,
                               4, 1,
                               192, 32, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_53
    CNN_MatAddAct_SQ8("S132_MatAdd_32x6x6", 0, 32, 6, 6, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S135_Conv2d_192x32x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S135_Conv2d_192x32x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S135_Conv2d_192x32x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_54_fusion
    CNN_ConvolutionPoolAct_SQ8("S135_Conv2d_192x32x1x1_Relu6", &gen_ctrl_S135_Conv2d_192x32x1x1_Relu6,
                               4, 1,
                               32, 192, 6, 6,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    CNN_GenControl_T gen_ctrl_S138_Conv2d_192x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S138_Conv2d_192x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S138_Conv2d_192x1x3x3_Relu6, "PADTYPE", AT_OPT_VAL(1));
    // generator for DEPTHWISE_CONV_2D_0_56_fusion
    CNN_ConvolutionPoolAct_SQ8("S138_Conv2d_192x1x3x3_Relu6", &gen_ctrl_S138_Conv2d_192x1x3x3_Relu6,
                               4, 1,
                               192, 192, 6, 6,
                               KOP_CONV_DW, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S141_Conv2d_56x192x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S141_Conv2d_56x192x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S141_Conv2d_56x192x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_57
    CNN_ConvolutionPoolAct_SQ8("S141_Conv2d_56x192x1x1", &gen_ctrl_S141_Conv2d_56x192x1x1,
                               4, 1,
                               192, 56, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S144_Conv2d_336x56x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S144_Conv2d_336x56x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S144_Conv2d_336x56x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_58_fusion
    CNN_ConvolutionPoolAct_SQ8("S144_Conv2d_336x56x1x1_Relu6", &gen_ctrl_S144_Conv2d_336x56x1x1_Relu6,
                               4, 1,
                               56, 336, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_59_fusion
    CNN_ConvolutionPoolAct_SQ8("S147_Conv2d_336x1x3x3_Relu6", 0,
                               4, 1,
                               336, 336, 3, 3,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S150_Conv2d_56x336x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S150_Conv2d_56x336x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S150_Conv2d_56x336x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_60
    CNN_ConvolutionPoolAct_SQ8("S150_Conv2d_56x336x1x1", &gen_ctrl_S150_Conv2d_56x336x1x1,
                               4, 1,
                               336, 56, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_61
    CNN_MatAddAct_SQ8("S151_MatAdd_56x3x3", 0, 56, 3, 3, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S154_Conv2d_336x56x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S154_Conv2d_336x56x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S154_Conv2d_336x56x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_62_fusion
    CNN_ConvolutionPoolAct_SQ8("S154_Conv2d_336x56x1x1_Relu6", &gen_ctrl_S154_Conv2d_336x56x1x1_Relu6,
                               4, 1,
                               56, 336, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_63_fusion
    CNN_ConvolutionPoolAct_SQ8("S157_Conv2d_336x1x3x3_Relu6", 0,
                               4, 1,
                               336, 336, 3, 3,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S160_Conv2d_56x336x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S160_Conv2d_56x336x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S160_Conv2d_56x336x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_64
    CNN_ConvolutionPoolAct_SQ8("S160_Conv2d_56x336x1x1", &gen_ctrl_S160_Conv2d_56x336x1x1,
                               4, 1,
                               336, 56, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    
    // generator for ADD_0_65
    CNN_MatAddAct_SQ8("S161_MatAdd_56x3x3", 0, 56, 3, 3, KOP_MATADD, KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S164_Conv2d_336x56x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S164_Conv2d_336x56x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S164_Conv2d_336x56x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_66_fusion
    CNN_ConvolutionPoolAct_SQ8("S164_Conv2d_336x56x1x1_Relu6", &gen_ctrl_S164_Conv2d_336x56x1x1_Relu6,
                               4, 1,
                               56, 336, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for DEPTHWISE_CONV_2D_0_67_fusion
    CNN_ConvolutionPoolAct_SQ8("S167_Conv2d_336x1x3x3_Relu6", 0,
                               4, 1,
                               336, 336, 3, 3,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    CNN_GenControl_T gen_ctrl_S170_Conv2d_112x336x1x1;
    CNN_InitGenCtrl(&gen_ctrl_S170_Conv2d_112x336x1x1);
    CNN_SetGenCtrl(&gen_ctrl_S170_Conv2d_112x336x1x1, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_68
    CNN_ConvolutionPoolAct_SQ8("S170_Conv2d_112x336x1x1", &gen_ctrl_S170_Conv2d_112x336x1x1,
                               4, 1,
                               336, 112, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S173_Conv2d_1280x112x1x1_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S173_Conv2d_1280x112x1x1_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S173_Conv2d_1280x112x1x1_Relu6, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_69_fusion
    CNN_ConvolutionPoolAct_SQ8("S173_Conv2d_1280x112x1x1_Relu6", &gen_ctrl_S173_Conv2d_1280x112x1x1_Relu6,
                               4, 1,
                               112, 1280, 3, 3,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    // generator for DEPTHWISE_CONV_2D_0_70
    CNN_ConvolutionPoolAct_SQ8("S176_Conv2d_1280x1x3x3", 0,
                               4, 1,
                               1280, 1280, 3, 3,
                               KOP_CONV_DW, 3, 3, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S179_Conv2d_32x1280x1x1_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S179_Conv2d_32x1280x1x1_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S179_Conv2d_32x1280x1x1_Relu, "ENABLEIM2COL", AT_OPT_VAL(1));
    // generator for CONV_2D_0_71_fusion
    CNN_ConvolutionPoolAct_SQ8("S179_Conv2d_32x1280x1x1_Relu", &gen_ctrl_S179_Conv2d_32x1280x1x1_Relu,
                               4, 1,
                               1280, 32, 1, 1,
                               KOP_CONV, 1, 1, 1, 1, 1, 1, 0,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);
    
    // generator for MEAN_0_72
    CNN_GlobalPoolAct_SQ8("S180_Op_MEAN_0_72", 0,
                          32, 1, 1,
                          KOP_GLOBAL_AVGPOOL, KOP_NONE);
    
    // generator for FULLY_CONNECTED_0_73
    CNN_LinearAct_SQ8("S183_Linear_2x32", 0,
                      4, 1,
                      32, 2,
                      KOP_LINEAR, KOP_NONE);
    
    // generator for SOFTMAX_0_74
    CNN_SoftMax_SQ8("S184_SoftMax", 0, 2, KOP_SOFTMAX);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("classificationCNN",
        /* Arguments either passed or globals */
            CArgs(299,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Separable_conv2ddepthwise_kern", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Separable_conv2ddepthwise_kern.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialseparable_conv2dsepa", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2dsepa.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S3_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialseparable_conv2dsepa_411470e1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2dsepa_411470e1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialseparable_conv2dbias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2dbias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S6_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96co", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96co.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bn", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bn.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96ex", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96ex.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96ex_2e8d685a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96ex_2e8d685a.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96ex_4dd58ac9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96ex_4dd58ac9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96ex_ec77a7bf", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96ex_ec77a7bf.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S16_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S16_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S16_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S16_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_ecf73e9c", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_ecf73e9c.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_d7b8fcc3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_d7b8fcc3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S19_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_5aec7246", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_5aec7246.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_96627e8b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_96627e8b.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S22_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S22_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S22_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_cb056cce", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_cb056cce.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_adb8c218", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_adb8c218.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S25_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S25_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S25_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S25_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S25_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_fc011502", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_fc011502.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_2be52376", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_2be52376.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S28_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S28_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S28_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S28_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S28_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_4d2ce9de", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_4d2ce9de.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_2448ea8a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_2448ea8a.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S31_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S31_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S31_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S31_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S31_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S31_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_9757a219", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9757a219.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_fc9e5e40", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_fc9e5e40.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S34_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S34_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S34_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S34_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S34_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S34_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [157] IN1SCALEN: [7] OUTSCALE: [67] OUTSCALEN: [6]
                TCArgInfo("signed char * __restrict__", "S35_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S35_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_4b74c721", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_4b74c721.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_9be0ce95", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9be0ce95.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S38_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S38_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S38_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S38_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S38_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S38_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_d944b225", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_d944b225.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_22a2b491", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_22a2b491.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S41_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S41_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S41_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S41_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S41_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S41_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_b7f5ec57", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_b7f5ec57.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_1f34f003", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_1f34f003.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S44_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S44_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S44_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S44_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S44_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S44_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_c6bf04ca", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_c6bf04ca.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_6437f573", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_6437f573.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S47_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S47_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S47_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S47_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S47_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S47_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_9fc268aa", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9fc268aa.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_11d2bf00", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_11d2bf00.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S50_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S50_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S50_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S50_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S50_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S50_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_8db0f664", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_8db0f664.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_f53b6b4d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_f53b6b4d.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S53_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S53_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S53_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S53_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S53_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S53_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [135] IN1SCALEN: [6] OUTSCALE: [203] OUTSCALEN: [9]
                TCArgInfo("signed char * __restrict__", "S54_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S54_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_f9b1c0db", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_f9b1c0db.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_c68d5f56", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_c68d5f56.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S57_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S57_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S57_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S57_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S57_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S57_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_886f09a8", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_886f09a8.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_dda18fbb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_dda18fbb.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S60_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S60_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S60_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S60_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S60_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S60_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_455dd475", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_455dd475.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_41e426e4", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_41e426e4.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S63_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S63_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S63_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S63_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S63_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S63_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [201] IN1SCALEN: [7] OUTSCALE: [251] OUTSCALEN: [9]
                TCArgInfo("signed char * __restrict__", "S64_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S64_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_d639dcef", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_d639dcef.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_cabdb5c1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_cabdb5c1.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S67_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S67_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S67_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S67_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S67_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S67_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_daa68995", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_daa68995.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_2cc947d0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_2cc947d0.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S70_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S70_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S70_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S70_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S70_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S70_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_000ffe14", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_000ffe14.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_bb7bc9fe", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_bb7bc9fe.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S73_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S73_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S73_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S73_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S73_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S73_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_b16f9f80", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_b16f9f80.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_64f865bc", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_64f865bc.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S76_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S76_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S76_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S76_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S76_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S76_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_7833e667", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_7833e667.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_81f4a85a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_81f4a85a.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S79_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S79_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S79_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S79_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S79_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S79_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_5a4b36c9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_5a4b36c9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_2764c46f", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_2764c46f.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S82_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S82_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S82_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S82_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S82_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S82_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [253] IN1SCALEN: [7] OUTSCALE: [31] OUTSCALEN: [6]
                TCArgInfo("signed char * __restrict__", "S83_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S83_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_3007959b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_3007959b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_5f1e7195", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_5f1e7195.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S86_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S86_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S86_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S86_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S86_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S86_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_cc4b30bd", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_cc4b30bd.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_83a3dc94", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_83a3dc94.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S89_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S89_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S89_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S89_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S89_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S89_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_9a51d69c", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9a51d69c.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_25106508", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_25106508.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S92_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S92_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S92_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S92_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S92_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S92_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [39] IN1SCALEN: [4] OUTSCALE: [57] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S93_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S93_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_9844f6db", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9844f6db.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_75209268", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_75209268.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S96_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S96_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S96_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S96_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S96_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S96_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_8f1ecd3b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_8f1ecd3b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_69b825c6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_69b825c6.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S99_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S99_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S99_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S99_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S99_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S99_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_266b0c79", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_266b0c79.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_84adc68c", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_84adc68c.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S102_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S102_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S102_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S102_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S102_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S102_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [23] IN1SCALEN: [4] OUTSCALE: [5] OUTSCALEN: [3]
                TCArgInfo("signed char * __restrict__", "S103_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S103_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_a9d3a918", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_a9d3a918.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_4e5c1323", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_4e5c1323.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S106_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S106_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S106_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S106_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S106_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S106_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_2fc4f9a9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_2fc4f9a9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_9b056684", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9b056684.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S109_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S109_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S109_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S109_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S109_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S109_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_915f853b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_915f853b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_c09767cb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_c09767cb.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S112_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S112_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S112_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S112_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S112_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S112_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_701cc74b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_701cc74b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_0d88a5ec", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_0d88a5ec.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S115_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S115_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S115_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S115_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04243 out: 0.04243  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S115_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S115_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_1aeb4567", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_1aeb4567.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_1a3894e6", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_1a3894e6.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S118_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S118_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S118_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S118_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S118_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S118_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_ec3a71e1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_ec3a71e1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_4a922648", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_4a922648.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S121_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S121_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S121_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S121_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S121_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S121_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [255] IN1SCALEN: [7] OUTSCALE: [99] OUTSCALEN: [8]
                TCArgInfo("signed char * __restrict__", "S122_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S122_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_83535371", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_83535371.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_f0f41adb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_f0f41adb.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S125_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S125_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S125_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S125_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S125_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S125_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_1d4d6859", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_1d4d6859.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_e1b513dd", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_e1b513dd.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S128_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S128_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S128_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S128_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S128_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S128_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_d6518a03", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_d6518a03.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_97ec88ff", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_97ec88ff.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S131_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S131_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S131_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S131_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S131_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S131_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [219] IN1SCALEN: [7] OUTSCALE: [79] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S132_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S132_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_5e6564a2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_5e6564a2.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_413d1607", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_413d1607.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S135_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S135_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S135_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S135_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03746 out: 0.03746  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S135_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S135_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_0d1bfdee", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_0d1bfdee.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_ce350816", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_ce350816.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S138_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S138_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S138_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S138_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S138_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S138_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_9b37a226", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_9b37a226.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_548058eb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_548058eb.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S141_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S141_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S141_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S141_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S141_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S141_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_666c577b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_666c577b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_51b7e479", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_51b7e479.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S144_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S144_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S144_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S144_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04270 out: 0.04270  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S144_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S144_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_3822911a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_3822911a.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_8af10a99", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_8af10a99.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S147_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S147_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S147_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S147_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S147_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S147_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_96b6fec7", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_96b6fec7.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_b4c31155", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_b4c31155.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S150_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S150_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S150_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S150_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S150_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S150_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [135] IN1SCALEN: [7] OUTSCALE: [99] OUTSCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S151_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S151_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_5073c59c", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_5073c59c.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_1be2d906", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_1be2d906.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S154_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S154_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S154_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S154_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.03997 out: 0.03997  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S154_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S154_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_f8cfb438", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_f8cfb438.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_00b159f0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_00b159f0.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S157_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S157_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S157_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S157_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S157_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S157_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_11f0296b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_11f0296b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_f3fc8846", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_f3fc8846.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S160_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S160_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S160_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S160_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S160_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S160_Infos.tensor", 1, 1, 8, 0)),
                // no activation -  IN1SCALE: [171] IN1SCALEN: [7] OUTSCALE: [51] OUTSCALEN: [6]
                TCArgInfo("signed char * __restrict__", "S161_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S161_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_6b6bef1a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_6b6bef1a.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_24e62b87", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_24e62b87.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S164_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S164_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S164_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S164_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S164_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S164_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_6f90883e", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_6f90883e.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_e9658647", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_e9658647.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S167_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S167_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S167_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S167_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S167_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S167_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96bl_52fbff0b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_52fbff0b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96bl_606f6059", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96bl_606f6059.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S170_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S170_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S170_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S170_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S170_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S170_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialmobilenetv2_035_96co_190efb1b", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96co_190efb1b.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialmobilenetv2_035_96co_06627b5d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialmobilenetv2_035_96co_06627b5d.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S173_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S173_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S173_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S173_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S173_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S173_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialseparable_conv2d_1se", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2d_1se.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Sequentialseparable_conv2d_1se_c86b8bf3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2d_1se_c86b8bf3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S176_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S176_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S176_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S176_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S176_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S176_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialseparable_conv2d_1se_6121c020", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialseparable_conv2d_1se_6121c020.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Separable_conv2d_1bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Separable_conv2d_1bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S179_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S179_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S179_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S179_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00114 out: 0.00114  actscale: [1] actscalen: [0] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S179_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S179_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: 0 ACTSCALEN: 0 GLOBAL_SUM_SCALE: [1] GLOBAL_SUM_SCALEN: [7]
                TCArgInfo("signed char * __restrict__", "S180_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S180_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialdensematmul", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialdensematmul.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Densebias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Densebias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S183_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S183_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S183_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S183_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S183_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S183_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.00098 out: 0.00003  BIASL_SM: [5]
                TCArgInfo("signed char * __restrict__", "S184_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S184_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed short * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(69,
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S31_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S34_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S35_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S38_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S41_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S44_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S47_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S50_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S53_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S57_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S60_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S63_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S64_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S67_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S70_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S73_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S76_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S79_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S82_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S83_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S86_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S89_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S92_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S93_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S96_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S99_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S102_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S103_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S106_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S109_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S112_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S115_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S118_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S121_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S122_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S125_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S128_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S131_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S132_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S135_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S138_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S141_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S144_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S147_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S150_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S151_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S154_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S157_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S160_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S161_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S164_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S167_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S170_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S173_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S176_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S179_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S180_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S183_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S3_Conv2d_1x1x1x1 inq 0.00<(i8--128.00)*1.00000000<255.00 weightsq -1.32<(i8-0.00)*0.01042988<1.32 outq -337.77<(i8-127.00)*1.32459462<0.00 biasesq -22397994.00<(i32-0.00)*0.01042988<22397993.99
    AddNode("S3_Conv2d_1x1x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Separable_conv2ddepthwise_kern", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2dsepa", 0),
            GNodeArg(GNA_OUT, "S3_Output", 0),
            GNodeArg(GNA_IN, "S3_Mul_scale", 0),
            GNodeArg(GNA_IN, "S3_Mul_shift", 0),
            GNodeArg(GNA_IN, "S3_Infos", 0)
        )
    );
    // Node S6_Conv2d_3x1x1x1 inq -337.77<(i8-127.00)*1.32459462<0.00 weightsq chan<(i8-0.00)*chan<chan outq -298.49<(i8-0.00)*2.33191998<296.15 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S6_Conv2d_3x1x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S3_Output", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2dsepa_411470e1", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2dbias", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0)
        )
    );
    // Node RESIZE_BILINEAR_0_3 inq -298.49<(i8-0.00)*2.33191998<296.15 forced outq -298.49<(i8-0.00)*2.33191998<296.15 forced
    AddNode("S7_Op_RESIZE_BILINEAR_0_3",
        Bindings(2,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0)
        )
    );
    // Node S10_Conv2d_16x3x3x3_Relu6 inq -298.49<(i8-0.00)*2.33191998<296.15 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S10_Conv2d_16x3x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96co", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bn", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node S13_Conv2d_16x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S13_Conv2d_16x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96ex", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96ex_2e8d685a", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node S16_Conv2d_8x16x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -65.42<(i8--12.00)*0.56396616<78.39 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S16_Conv2d_8x16x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96ex_4dd58ac9", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96ex_ec77a7bf", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0),
            GNodeArg(GNA_IN, "S16_Mul_scale", 0),
            GNodeArg(GNA_IN, "S16_Mul_shift", 0),
            GNodeArg(GNA_IN, "S16_Infos", 0)
        )
    );
    // Node S19_Conv2d_48x8x1x1_Relu6 inq -65.42<(i8--12.00)*0.56396616<78.39 weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S19_Conv2d_48x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S16_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_ecf73e9c", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_d7b8fcc3", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0),
            GNodeArg(GNA_IN, "S19_Mul_scale", 0),
            GNodeArg(GNA_IN, "S19_Mul_shift", 0),
            GNodeArg(GNA_IN, "S19_Infos", 0)
        )
    );
    // Node S22_Conv2d_48x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S22_Conv2d_48x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_5aec7246", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_96627e8b", 0),
            GNodeArg(GNA_OUT, "S22_Output", 0),
            GNodeArg(GNA_IN, "S22_Mul_scale", 0),
            GNodeArg(GNA_IN, "S22_Mul_shift", 0),
            GNodeArg(GNA_IN, "S22_Infos", 0)
        )
    );
    // Node S25_Conv2d_8x48x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -73.59<(i8-0.00)*0.57492447<73.02 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S25_Conv2d_8x48x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S22_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_cb056cce", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_adb8c218", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0),
            GNodeArg(GNA_IN, "S25_Mul_scale", 0),
            GNodeArg(GNA_IN, "S25_Mul_shift", 0),
            GNodeArg(GNA_IN, "S25_Infos", 0)
        )
    );
    // Node S28_Conv2d_48x8x1x1_Relu6 inq -73.59<(i8-0.00)*0.57492447<73.02 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S28_Conv2d_48x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_fc011502", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_2be52376", 0),
            GNodeArg(GNA_OUT, "S28_Output", 0),
            GNodeArg(GNA_IN, "S28_Mul_scale", 0),
            GNodeArg(GNA_IN, "S28_Mul_shift", 0),
            GNodeArg(GNA_IN, "S28_Infos", 0)
        )
    );
    // Node S31_Conv2d_48x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S31_Conv2d_48x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_4d2ce9de", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_2448ea8a", 0),
            GNodeArg(GNA_OUT, "S31_Output", 0),
            GNodeArg(GNA_IN, "S31_Mul_scale", 0),
            GNodeArg(GNA_IN, "S31_Mul_shift", 0),
            GNodeArg(GNA_IN, "S31_Infos", 0)
        )
    );
    // Node S34_Conv2d_8x48x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -90.54<(i8-0.00)*0.70733715<89.83 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S34_Conv2d_8x48x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S31_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9757a219", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_fc9e5e40", 0),
            GNodeArg(GNA_OUT, "S34_Output", 0),
            GNodeArg(GNA_IN, "S34_Mul_scale", 0),
            GNodeArg(GNA_IN, "S34_Mul_shift", 0),
            GNodeArg(GNA_IN, "S34_Infos", 0)
        )
    );
    // Node S35_MatAdd_8x24x24 in1q -90.54<(i8-0.00)*0.70733715<89.83 forced
    //   in2q -73.59<(i8-0.00)*0.57492447<73.02 forced
    //   outq -70.18<(i8-0.00)*0.54831087<69.64 forced scaled input 0 is node input 1
    AddNode("S35_MatAdd_8x24x24",
        Bindings(4,
            GNodeArg(GNA_IN, "S34_Output", 0),
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_OUT, "S35_Output", 0),
            GNodeArg(GNA_IN, "S35_Infos", 0)
        )
    );
    // Node S38_Conv2d_48x8x1x1_Relu6 inq -70.18<(i8-0.00)*0.54831087<69.64 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S38_Conv2d_48x8x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S35_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_4b74c721", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9be0ce95", 0),
            GNodeArg(GNA_OUT, "S38_Output", 0),
            GNodeArg(GNA_IN, "S38_Mul_scale", 0),
            GNodeArg(GNA_IN, "S38_Mul_shift", 0),
            GNodeArg(GNA_IN, "S38_Infos", 0)
        )
    );
    // Node S41_Conv2d_48x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S41_Conv2d_48x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S38_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_d944b225", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_22a2b491", 0),
            GNodeArg(GNA_OUT, "S41_Output", 0),
            GNodeArg(GNA_IN, "S41_Mul_scale", 0),
            GNodeArg(GNA_IN, "S41_Mul_shift", 0),
            GNodeArg(GNA_IN, "S41_Infos", 0)
        )
    );
    // Node S44_Conv2d_16x48x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -46.07<(i8-0.00)*0.35991311<45.71 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S44_Conv2d_16x48x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S41_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_b7f5ec57", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_1f34f003", 0),
            GNodeArg(GNA_OUT, "S44_Output", 0),
            GNodeArg(GNA_IN, "S44_Mul_scale", 0),
            GNodeArg(GNA_IN, "S44_Mul_shift", 0),
            GNodeArg(GNA_IN, "S44_Infos", 0)
        )
    );
    // Node S47_Conv2d_96x16x1x1_Relu6 inq -46.07<(i8-0.00)*0.35991311<45.71 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S47_Conv2d_96x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S44_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_c6bf04ca", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_6437f573", 0),
            GNodeArg(GNA_OUT, "S47_Output", 0),
            GNodeArg(GNA_IN, "S47_Mul_scale", 0),
            GNodeArg(GNA_IN, "S47_Mul_shift", 0),
            GNodeArg(GNA_IN, "S47_Infos", 0)
        )
    );
    // Node S50_Conv2d_96x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S50_Conv2d_96x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S47_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9fc268aa", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_11d2bf00", 0),
            GNodeArg(GNA_OUT, "S50_Output", 0),
            GNodeArg(GNA_IN, "S50_Mul_scale", 0),
            GNodeArg(GNA_IN, "S50_Mul_shift", 0),
            GNodeArg(GNA_IN, "S50_Infos", 0)
        )
    );
    // Node S53_Conv2d_16x96x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -21.85<(i8-0.00)*0.17073422<21.68 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S53_Conv2d_16x96x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S50_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_8db0f664", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_f53b6b4d", 0),
            GNodeArg(GNA_OUT, "S53_Output", 0),
            GNodeArg(GNA_IN, "S53_Mul_scale", 0),
            GNodeArg(GNA_IN, "S53_Mul_shift", 0),
            GNodeArg(GNA_IN, "S53_Infos", 0)
        )
    );
    // Node S54_MatAdd_16x12x12 in1q -46.07<(i8-0.00)*0.35991311<45.71 forced
    //   in2q -21.85<(i8-0.00)*0.17073422<21.68 forced
    //   outq -55.08<(i8-0.00)*0.43031542<54.65 forced scaled input 0 is node input 0
    AddNode("S54_MatAdd_16x12x12",
        Bindings(4,
            GNodeArg(GNA_IN, "S44_Output", 0),
            GNodeArg(GNA_IN, "S53_Output", 0),
            GNodeArg(GNA_OUT, "S54_Output", 0),
            GNodeArg(GNA_IN, "S54_Infos", 0)
        )
    );
    // Node S57_Conv2d_96x16x1x1_Relu6 inq -55.08<(i8-0.00)*0.43031542<54.65 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S57_Conv2d_96x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_f9b1c0db", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_c68d5f56", 0),
            GNodeArg(GNA_OUT, "S57_Output", 0),
            GNodeArg(GNA_IN, "S57_Mul_scale", 0),
            GNodeArg(GNA_IN, "S57_Mul_shift", 0),
            GNodeArg(GNA_IN, "S57_Infos", 0)
        )
    );
    // Node S60_Conv2d_96x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S60_Conv2d_96x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S57_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_886f09a8", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_dda18fbb", 0),
            GNodeArg(GNA_OUT, "S60_Output", 0),
            GNodeArg(GNA_IN, "S60_Mul_scale", 0),
            GNodeArg(GNA_IN, "S60_Mul_shift", 0),
            GNodeArg(GNA_IN, "S60_Infos", 0)
        )
    );
    // Node S63_Conv2d_16x96x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -35.02<(i8-0.00)*0.27358118<34.74 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S63_Conv2d_16x96x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S60_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_455dd475", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_41e426e4", 0),
            GNodeArg(GNA_OUT, "S63_Output", 0),
            GNodeArg(GNA_IN, "S63_Mul_scale", 0),
            GNodeArg(GNA_IN, "S63_Mul_shift", 0),
            GNodeArg(GNA_IN, "S63_Infos", 0)
        )
    );
    // Node S64_MatAdd_16x12x12 in1q -55.08<(i8-0.00)*0.43031542<54.65 forced
    //   in2q -35.02<(i8-0.00)*0.27358118<34.74 forced
    //   outq -71.45<(i8-0.00)*0.55821602<70.89 forced scaled input 0 is node input 0
    AddNode("S64_MatAdd_16x12x12",
        Bindings(4,
            GNodeArg(GNA_IN, "S54_Output", 0),
            GNodeArg(GNA_IN, "S63_Output", 0),
            GNodeArg(GNA_OUT, "S64_Output", 0),
            GNodeArg(GNA_IN, "S64_Infos", 0)
        )
    );
    // Node S67_Conv2d_96x16x1x1_Relu6 inq -71.45<(i8-0.00)*0.55821602<70.89 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S67_Conv2d_96x16x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S64_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_d639dcef", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_cabdb5c1", 0),
            GNodeArg(GNA_OUT, "S67_Output", 0),
            GNodeArg(GNA_IN, "S67_Mul_scale", 0),
            GNodeArg(GNA_IN, "S67_Mul_shift", 0),
            GNodeArg(GNA_IN, "S67_Infos", 0)
        )
    );
    // Node S70_Conv2d_96x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S70_Conv2d_96x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S67_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_daa68995", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_2cc947d0", 0),
            GNodeArg(GNA_OUT, "S70_Output", 0),
            GNodeArg(GNA_IN, "S70_Mul_scale", 0),
            GNodeArg(GNA_IN, "S70_Mul_shift", 0),
            GNodeArg(GNA_IN, "S70_Infos", 0)
        )
    );
    // Node S73_Conv2d_24x96x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -43.90<(i8-0.00)*0.34293210<43.55 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S73_Conv2d_24x96x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S70_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_000ffe14", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_bb7bc9fe", 0),
            GNodeArg(GNA_OUT, "S73_Output", 0),
            GNodeArg(GNA_IN, "S73_Mul_scale", 0),
            GNodeArg(GNA_IN, "S73_Mul_shift", 0),
            GNodeArg(GNA_IN, "S73_Infos", 0)
        )
    );
    // Node S76_Conv2d_144x24x1x1_Relu6 inq -43.90<(i8-0.00)*0.34293210<43.55 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S76_Conv2d_144x24x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S73_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_b16f9f80", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_64f865bc", 0),
            GNodeArg(GNA_OUT, "S76_Output", 0),
            GNodeArg(GNA_IN, "S76_Mul_scale", 0),
            GNodeArg(GNA_IN, "S76_Mul_shift", 0),
            GNodeArg(GNA_IN, "S76_Infos", 0)
        )
    );
    // Node S79_Conv2d_144x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S79_Conv2d_144x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S76_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_7833e667", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_81f4a85a", 0),
            GNodeArg(GNA_OUT, "S79_Output", 0),
            GNodeArg(GNA_IN, "S79_Mul_scale", 0),
            GNodeArg(GNA_IN, "S79_Mul_shift", 0),
            GNodeArg(GNA_IN, "S79_Infos", 0)
        )
    );
    // Node S82_Conv2d_24x144x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -22.20<(i8-0.00)*0.17346883<22.03 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S82_Conv2d_24x144x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S79_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_5a4b36c9", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_2764c46f", 0),
            GNodeArg(GNA_OUT, "S82_Output", 0),
            GNodeArg(GNA_IN, "S82_Mul_scale", 0),
            GNodeArg(GNA_IN, "S82_Mul_shift", 0),
            GNodeArg(GNA_IN, "S82_Infos", 0)
        )
    );
    // Node S83_MatAdd_24x6x6 in1q -43.90<(i8-0.00)*0.34293210<43.55 forced
    //   in2q -22.20<(i8-0.00)*0.17346883<22.03 forced
    //   outq -45.79<(i8-0.00)*0.35776053<45.44 forced scaled input 0 is node input 0
    AddNode("S83_MatAdd_24x6x6",
        Bindings(4,
            GNodeArg(GNA_IN, "S73_Output", 0),
            GNodeArg(GNA_IN, "S82_Output", 0),
            GNodeArg(GNA_OUT, "S83_Output", 0),
            GNodeArg(GNA_IN, "S83_Infos", 0)
        )
    );
    // Node S86_Conv2d_144x24x1x1_Relu6 inq -45.79<(i8-0.00)*0.35776053<45.44 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S86_Conv2d_144x24x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S83_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_3007959b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_5f1e7195", 0),
            GNodeArg(GNA_OUT, "S86_Output", 0),
            GNodeArg(GNA_IN, "S86_Mul_scale", 0),
            GNodeArg(GNA_IN, "S86_Mul_shift", 0),
            GNodeArg(GNA_IN, "S86_Infos", 0)
        )
    );
    // Node S89_Conv2d_144x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S89_Conv2d_144x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S86_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_cc4b30bd", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_83a3dc94", 0),
            GNodeArg(GNA_OUT, "S89_Output", 0),
            GNodeArg(GNA_IN, "S89_Mul_scale", 0),
            GNodeArg(GNA_IN, "S89_Mul_shift", 0),
            GNodeArg(GNA_IN, "S89_Infos", 0)
        )
    );
    // Node S92_Conv2d_24x144x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -18.77<(i8-0.00)*0.14664367<18.62 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S92_Conv2d_24x144x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S89_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9a51d69c", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_25106508", 0),
            GNodeArg(GNA_OUT, "S92_Output", 0),
            GNodeArg(GNA_IN, "S92_Mul_scale", 0),
            GNodeArg(GNA_IN, "S92_Mul_shift", 0),
            GNodeArg(GNA_IN, "S92_Infos", 0)
        )
    );
    // Node S93_MatAdd_24x6x6 in1q -45.79<(i8-0.00)*0.35776053<45.44 forced
    //   in2q -18.77<(i8-0.00)*0.14664367<18.62 forced
    //   outq -42.24<(i8-0.00)*0.32999987<41.91 forced scaled input 0 is node input 0
    AddNode("S93_MatAdd_24x6x6",
        Bindings(4,
            GNodeArg(GNA_IN, "S83_Output", 0),
            GNodeArg(GNA_IN, "S92_Output", 0),
            GNodeArg(GNA_OUT, "S93_Output", 0),
            GNodeArg(GNA_IN, "S93_Infos", 0)
        )
    );
    // Node S96_Conv2d_144x24x1x1_Relu6 inq -42.24<(i8-0.00)*0.32999987<41.91 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S96_Conv2d_144x24x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S93_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9844f6db", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_75209268", 0),
            GNodeArg(GNA_OUT, "S96_Output", 0),
            GNodeArg(GNA_IN, "S96_Mul_scale", 0),
            GNodeArg(GNA_IN, "S96_Mul_shift", 0),
            GNodeArg(GNA_IN, "S96_Infos", 0)
        )
    );
    // Node S99_Conv2d_144x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S99_Conv2d_144x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S96_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_8f1ecd3b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_69b825c6", 0),
            GNodeArg(GNA_OUT, "S99_Output", 0),
            GNodeArg(GNA_IN, "S99_Mul_scale", 0),
            GNodeArg(GNA_IN, "S99_Mul_shift", 0),
            GNodeArg(GNA_IN, "S99_Infos", 0)
        )
    );
    // Node S102_Conv2d_24x144x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -29.44<(i8-0.00)*0.22999526<29.21 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S102_Conv2d_24x144x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S99_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_266b0c79", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_84adc68c", 0),
            GNodeArg(GNA_OUT, "S102_Output", 0),
            GNodeArg(GNA_IN, "S102_Mul_scale", 0),
            GNodeArg(GNA_IN, "S102_Mul_shift", 0),
            GNodeArg(GNA_IN, "S102_Infos", 0)
        )
    );
    // Node S103_MatAdd_24x6x6 in1q -42.24<(i8-0.00)*0.32999987<41.91 forced
    //   in2q -29.44<(i8-0.00)*0.22999526<29.21 forced
    //   outq -47.04<(i8-0.00)*0.36747774<46.67 forced scaled input 0 is node input 0
    AddNode("S103_MatAdd_24x6x6",
        Bindings(4,
            GNodeArg(GNA_IN, "S93_Output", 0),
            GNodeArg(GNA_IN, "S102_Output", 0),
            GNodeArg(GNA_OUT, "S103_Output", 0),
            GNodeArg(GNA_IN, "S103_Infos", 0)
        )
    );
    // Node S106_Conv2d_144x24x1x1_Relu6 inq -47.04<(i8-0.00)*0.36747774<46.67 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S106_Conv2d_144x24x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S103_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_a9d3a918", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_4e5c1323", 0),
            GNodeArg(GNA_OUT, "S106_Output", 0),
            GNodeArg(GNA_IN, "S106_Mul_scale", 0),
            GNodeArg(GNA_IN, "S106_Mul_shift", 0),
            GNodeArg(GNA_IN, "S106_Infos", 0)
        )
    );
    // Node S109_Conv2d_144x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S109_Conv2d_144x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S106_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_2fc4f9a9", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9b056684", 0),
            GNodeArg(GNA_OUT, "S109_Output", 0),
            GNodeArg(GNA_IN, "S109_Mul_scale", 0),
            GNodeArg(GNA_IN, "S109_Mul_shift", 0),
            GNodeArg(GNA_IN, "S109_Infos", 0)
        )
    );
    // Node S112_Conv2d_32x144x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -23.48<(i8-0.00)*0.18342851<23.30 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S112_Conv2d_32x144x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S109_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_915f853b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_c09767cb", 0),
            GNodeArg(GNA_OUT, "S112_Output", 0),
            GNodeArg(GNA_IN, "S112_Mul_scale", 0),
            GNodeArg(GNA_IN, "S112_Mul_shift", 0),
            GNodeArg(GNA_IN, "S112_Infos", 0)
        )
    );
    // Node S115_Conv2d_192x32x1x1_Relu6 inq -23.48<(i8-0.00)*0.18342851<23.30 forced weightsq chan<(i8-0.00)*chan<chan outq -5.43<(i8-0.00)*0.04243240<5.39 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S115_Conv2d_192x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S112_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_701cc74b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_0d88a5ec", 0),
            GNodeArg(GNA_OUT, "S115_Output", 0),
            GNodeArg(GNA_IN, "S115_Mul_scale", 0),
            GNodeArg(GNA_IN, "S115_Mul_shift", 0),
            GNodeArg(GNA_IN, "S115_Infos", 0)
        )
    );
    // Node S118_Conv2d_192x1x3x3_Relu6 inq -5.43<(i8-0.00)*0.04243240<5.39 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S118_Conv2d_192x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S115_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_1aeb4567", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_1a3894e6", 0),
            GNodeArg(GNA_OUT, "S118_Output", 0),
            GNodeArg(GNA_IN, "S118_Mul_scale", 0),
            GNodeArg(GNA_IN, "S118_Mul_shift", 0),
            GNodeArg(GNA_IN, "S118_Infos", 0)
        )
    );
    // Node S121_Conv2d_32x192x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -11.77<(i8-0.00)*0.09194992<11.68 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S121_Conv2d_32x192x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S118_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_ec3a71e1", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_4a922648", 0),
            GNodeArg(GNA_OUT, "S121_Output", 0),
            GNodeArg(GNA_IN, "S121_Mul_scale", 0),
            GNodeArg(GNA_IN, "S121_Mul_shift", 0),
            GNodeArg(GNA_IN, "S121_Infos", 0)
        )
    );
    // Node S122_MatAdd_32x6x6 in1q -23.48<(i8-0.00)*0.18342851<23.30 forced
    //   in2q -11.77<(i8-0.00)*0.09194992<11.68 forced
    //   outq -30.48<(i8-0.00)*0.23815272<30.25 forced scaled input 0 is node input 0
    AddNode("S122_MatAdd_32x6x6",
        Bindings(4,
            GNodeArg(GNA_IN, "S112_Output", 0),
            GNodeArg(GNA_IN, "S121_Output", 0),
            GNodeArg(GNA_OUT, "S122_Output", 0),
            GNodeArg(GNA_IN, "S122_Infos", 0)
        )
    );
    // Node S125_Conv2d_192x32x1x1_Relu6 inq -30.48<(i8-0.00)*0.23815272<30.25 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S125_Conv2d_192x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_83535371", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_f0f41adb", 0),
            GNodeArg(GNA_OUT, "S125_Output", 0),
            GNodeArg(GNA_IN, "S125_Mul_scale", 0),
            GNodeArg(GNA_IN, "S125_Mul_shift", 0),
            GNodeArg(GNA_IN, "S125_Infos", 0)
        )
    );
    // Node S128_Conv2d_192x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S128_Conv2d_192x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S125_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_1d4d6859", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_e1b513dd", 0),
            GNodeArg(GNA_OUT, "S128_Output", 0),
            GNodeArg(GNA_IN, "S128_Mul_scale", 0),
            GNodeArg(GNA_IN, "S128_Mul_shift", 0),
            GNodeArg(GNA_IN, "S128_Infos", 0)
        )
    );
    // Node S131_Conv2d_32x192x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -17.78<(i8-0.00)*0.13889682<17.64 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S131_Conv2d_32x192x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S128_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_d6518a03", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_97ec88ff", 0),
            GNodeArg(GNA_OUT, "S131_Output", 0),
            GNodeArg(GNA_IN, "S131_Mul_scale", 0),
            GNodeArg(GNA_IN, "S131_Mul_shift", 0),
            GNodeArg(GNA_IN, "S131_Infos", 0)
        )
    );
    // Node S132_MatAdd_32x6x6 in1q -30.48<(i8-0.00)*0.23815272<30.25 forced
    //   in2q -17.78<(i8-0.00)*0.13889682<17.64 forced
    //   outq -28.72<(i8-0.00)*0.22439858<28.50 forced scaled input 0 is node input 0
    AddNode("S132_MatAdd_32x6x6",
        Bindings(4,
            GNodeArg(GNA_IN, "S122_Output", 0),
            GNodeArg(GNA_IN, "S131_Output", 0),
            GNodeArg(GNA_OUT, "S132_Output", 0),
            GNodeArg(GNA_IN, "S132_Infos", 0)
        )
    );
    // Node S135_Conv2d_192x32x1x1_Relu6 inq -28.72<(i8-0.00)*0.22439858<28.50 forced weightsq chan<(i8-0.00)*chan<chan outq -4.79<(i8-0.00)*0.03745705<4.76 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S135_Conv2d_192x32x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S132_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_5e6564a2", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_413d1607", 0),
            GNodeArg(GNA_OUT, "S135_Output", 0),
            GNodeArg(GNA_IN, "S135_Mul_scale", 0),
            GNodeArg(GNA_IN, "S135_Mul_shift", 0),
            GNodeArg(GNA_IN, "S135_Infos", 0)
        )
    );
    // Node S138_Conv2d_192x1x3x3_Relu6 inq -4.79<(i8-0.00)*0.03745705<4.76 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S138_Conv2d_192x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S135_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_0d1bfdee", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_ce350816", 0),
            GNodeArg(GNA_OUT, "S138_Output", 0),
            GNodeArg(GNA_IN, "S138_Mul_scale", 0),
            GNodeArg(GNA_IN, "S138_Mul_shift", 0),
            GNodeArg(GNA_IN, "S138_Infos", 0)
        )
    );
    // Node S141_Conv2d_56x192x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -16.26<(i8-0.00)*0.12702124<16.13 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S141_Conv2d_56x192x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S138_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_9b37a226", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_548058eb", 0),
            GNodeArg(GNA_OUT, "S141_Output", 0),
            GNodeArg(GNA_IN, "S141_Mul_scale", 0),
            GNodeArg(GNA_IN, "S141_Mul_shift", 0),
            GNodeArg(GNA_IN, "S141_Infos", 0)
        )
    );
    // Node S144_Conv2d_336x56x1x1_Relu6 inq -16.26<(i8-0.00)*0.12702124<16.13 forced weightsq chan<(i8-0.00)*chan<chan outq -5.47<(i8-0.00)*0.04269575<5.42 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S144_Conv2d_336x56x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_666c577b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_51b7e479", 0),
            GNodeArg(GNA_OUT, "S144_Output", 0),
            GNodeArg(GNA_IN, "S144_Mul_scale", 0),
            GNodeArg(GNA_IN, "S144_Mul_shift", 0),
            GNodeArg(GNA_IN, "S144_Infos", 0)
        )
    );
    // Node S147_Conv2d_336x1x3x3_Relu6 inq -5.47<(i8-0.00)*0.04269575<5.42 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S147_Conv2d_336x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S144_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_3822911a", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_8af10a99", 0),
            GNodeArg(GNA_OUT, "S147_Output", 0),
            GNodeArg(GNA_IN, "S147_Mul_scale", 0),
            GNodeArg(GNA_IN, "S147_Mul_shift", 0),
            GNodeArg(GNA_IN, "S147_Infos", 0)
        )
    );
    // Node S150_Conv2d_56x336x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -15.47<(i8-0.00)*0.12082767<15.35 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S150_Conv2d_56x336x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S147_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_96b6fec7", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_b4c31155", 0),
            GNodeArg(GNA_OUT, "S150_Output", 0),
            GNodeArg(GNA_IN, "S150_Mul_scale", 0),
            GNodeArg(GNA_IN, "S150_Mul_shift", 0),
            GNodeArg(GNA_IN, "S150_Infos", 0)
        )
    );
    // Node S151_MatAdd_56x3x3 in1q -16.26<(i8-0.00)*0.12702124<16.13 forced
    //   in2q -15.47<(i8-0.00)*0.12082767<15.35 forced
    //   outq -19.95<(i8-0.00)*0.15585626<19.79 forced scaled input 0 is node input 0
    AddNode("S151_MatAdd_56x3x3",
        Bindings(4,
            GNodeArg(GNA_IN, "S141_Output", 0),
            GNodeArg(GNA_IN, "S150_Output", 0),
            GNodeArg(GNA_OUT, "S151_Output", 0),
            GNodeArg(GNA_IN, "S151_Infos", 0)
        )
    );
    // Node S154_Conv2d_336x56x1x1_Relu6 inq -19.95<(i8-0.00)*0.15585626<19.79 forced weightsq chan<(i8-0.00)*chan<chan outq -5.12<(i8-0.00)*0.03997144<5.08 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S154_Conv2d_336x56x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S151_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_5073c59c", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_1be2d906", 0),
            GNodeArg(GNA_OUT, "S154_Output", 0),
            GNodeArg(GNA_IN, "S154_Mul_scale", 0),
            GNodeArg(GNA_IN, "S154_Mul_shift", 0),
            GNodeArg(GNA_IN, "S154_Infos", 0)
        )
    );
    // Node S157_Conv2d_336x1x3x3_Relu6 inq -5.12<(i8-0.00)*0.03997144<5.08 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S157_Conv2d_336x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S154_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_f8cfb438", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_00b159f0", 0),
            GNodeArg(GNA_OUT, "S157_Output", 0),
            GNodeArg(GNA_IN, "S157_Mul_scale", 0),
            GNodeArg(GNA_IN, "S157_Mul_shift", 0),
            GNodeArg(GNA_IN, "S157_Infos", 0)
        )
    );
    // Node S160_Conv2d_56x336x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -14.96<(i8-0.00)*0.11686135<14.84 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S160_Conv2d_56x336x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S157_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_11f0296b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_f3fc8846", 0),
            GNodeArg(GNA_OUT, "S160_Output", 0),
            GNodeArg(GNA_IN, "S160_Mul_scale", 0),
            GNodeArg(GNA_IN, "S160_Mul_shift", 0),
            GNodeArg(GNA_IN, "S160_Infos", 0)
        )
    );
    // Node S161_MatAdd_56x3x3 in1q -19.95<(i8-0.00)*0.15585626<19.79 forced
    //   in2q -14.96<(i8-0.00)*0.11686135<14.84 forced
    //   outq -18.74<(i8-0.00)*0.14639721<18.59 forced scaled input 0 is node input 0
    AddNode("S161_MatAdd_56x3x3",
        Bindings(4,
            GNodeArg(GNA_IN, "S151_Output", 0),
            GNodeArg(GNA_IN, "S160_Output", 0),
            GNodeArg(GNA_OUT, "S161_Output", 0),
            GNodeArg(GNA_IN, "S161_Infos", 0)
        )
    );
    // Node S164_Conv2d_336x56x1x1_Relu6 inq -18.74<(i8-0.00)*0.14639721<18.59 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724410<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S164_Conv2d_336x56x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S161_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_6b6bef1a", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_24e62b87", 0),
            GNodeArg(GNA_OUT, "S164_Output", 0),
            GNodeArg(GNA_IN, "S164_Mul_scale", 0),
            GNodeArg(GNA_IN, "S164_Mul_shift", 0),
            GNodeArg(GNA_IN, "S164_Infos", 0)
        )
    );
    // Node S167_Conv2d_336x1x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724410<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S167_Conv2d_336x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S164_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_6f90883e", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_e9658647", 0),
            GNodeArg(GNA_OUT, "S167_Output", 0),
            GNodeArg(GNA_IN, "S167_Mul_scale", 0),
            GNodeArg(GNA_IN, "S167_Mul_shift", 0),
            GNodeArg(GNA_IN, "S167_Infos", 0)
        )
    );
    // Node S170_Conv2d_112x336x1x1 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -8.66<(i8-1.00)*0.06713483<8.46 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S170_Conv2d_112x336x1x1",
        Bindings(7,
            GNodeArg(GNA_IN, "S167_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_52fbff0b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96bl_606f6059", 0),
            GNodeArg(GNA_OUT, "S170_Output", 0),
            GNodeArg(GNA_IN, "S170_Mul_scale", 0),
            GNodeArg(GNA_IN, "S170_Mul_shift", 0),
            GNodeArg(GNA_IN, "S170_Infos", 0)
        )
    );
    // Node S173_Conv2d_1280x112x1x1_Relu6 inq -8.66<(i8-1.00)*0.06713483<8.46 weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S173_Conv2d_1280x112x1x1_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S170_Output", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96co_190efb1b", 0),
            GNodeArg(GNA_IN, "Sequentialmobilenetv2_035_96co_06627b5d", 0),
            GNodeArg(GNA_OUT, "S173_Output", 0),
            GNodeArg(GNA_IN, "S173_Mul_scale", 0),
            GNodeArg(GNA_IN, "S173_Mul_shift", 0),
            GNodeArg(GNA_IN, "S173_Infos", 0)
        )
    );
    // Node S176_Conv2d_1280x1x3x3 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq chan<(i8-0.00)*chan<chan outq -0.36<(i8-0.00)*0.00279861<0.36 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S176_Conv2d_1280x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S173_Output", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2d_1se", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2d_1se_c86b8bf3", 0),
            GNodeArg(GNA_OUT, "S176_Output", 0),
            GNodeArg(GNA_IN, "S176_Mul_scale", 0),
            GNodeArg(GNA_IN, "S176_Mul_shift", 0),
            GNodeArg(GNA_IN, "S176_Infos", 0)
        )
    );
    // Node S179_Conv2d_32x1280x1x1_Relu inq -0.36<(i8-0.00)*0.00279861<0.36 weightsq chan<(i8-0.00)*chan<chan outq -0.15<(i8-0.00)*0.00113869<0.14 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("S179_Conv2d_32x1280x1x1_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S176_Output", 0),
            GNodeArg(GNA_IN, "Sequentialseparable_conv2d_1se_6121c020", 0),
            GNodeArg(GNA_IN, "Separable_conv2d_1bias", 0),
            GNodeArg(GNA_OUT, "S179_Output", 0),
            GNodeArg(GNA_IN, "S179_Mul_scale", 0),
            GNodeArg(GNA_IN, "S179_Mul_shift", 0),
            GNodeArg(GNA_IN, "S179_Infos", 0)
        )
    );
    // Node MEAN_0_72 inq -0.15<(i8-0.00)*0.00113869<0.14 forced outq -0.15<(i8-0.00)*0.00113869<0.14
    AddNode("S180_Op_MEAN_0_72",
        Bindings(3,
            GNodeArg(GNA_IN, "S179_Output", 0),
            GNodeArg(GNA_OUT, "S180_Output", 0),
            GNodeArg(GNA_IN, "S180_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_73 inq -0.15<(i8-0.00)*0.00113869<0.14 weightsq chan<(i8-0.00)*chan<chan outq -0.12<(i8-0.00)*0.00097656<0.12 forced
    AddNode("S183_Linear_2x32",
        Bindings(7,
            GNodeArg(GNA_IN, "S180_Output", 0),
            GNodeArg(GNA_IN, "Sequentialdensematmul", 0),
            GNodeArg(GNA_IN, "Densebias", 0),
            GNodeArg(GNA_OUT, "S183_Output", 0),
            GNodeArg(GNA_IN, "S183_Mul_scale", 0),
            GNodeArg(GNA_IN, "S183_Mul_shift", 0),
            GNodeArg(GNA_IN, "S183_Infos", 0)
        )
    );
    // Node SOFTMAX_0_74 inq -0.12<(i8-0.00)*0.00097656<0.12 forced outq -1.00<(i16-0.00)*0.00003052<1.00
    AddNode("S184_SoftMax",
        Bindings(3,
            GNodeArg(GNA_IN, "S183_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S184_Infos", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    classificationModel(64000, 300000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
