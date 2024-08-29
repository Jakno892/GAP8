#ifndef __CLASSIFICATIONKERNEL_H__
#define __CLASSIFICATIONKERNEL_H__

#include "AutoTilerLibTypes.h"
#include "Gap.h"
#include "classification.h"
#include "CNN_BasicKernels_SQ8.h"
#include "ResizeBasicKernels.h"
#define _classification_L1_Memory_SIZE 46696
#define _classification_L2_Memory_SIZE 76208
#define _classification_L2_Memory_Dyn_SIZE 193792
extern char *classification_L1_Memory; /* Size given for generation: 46736 bytes, used: 46696 bytes */
extern char *classification_L2_Memory; /* Size used for generation (static): 76208 bytes */
extern char *classification_L2_Memory_Dyn; /* Size used for generation (dynamic): 193792 bytes */
extern void S3_Conv2d_1x1x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S6_Conv2d_3x1x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S7_Op_RESIZE_BILINEAR_0_3(
		signed char * In,
		signed char * Out);
extern void S10_Conv2d_16x3x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S13_Conv2d_16x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S16_Conv2d_8x16x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S19_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S22_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S25_Conv2d_8x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S28_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S31_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S34_Conv2d_8x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S35_MatAdd_8x24x24(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S38_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S41_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S44_Conv2d_16x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S47_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S50_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S53_Conv2d_16x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S54_MatAdd_16x12x12(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S57_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S60_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S63_Conv2d_16x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S64_MatAdd_16x12x12(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S67_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S70_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S73_Conv2d_24x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S76_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S79_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S82_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S83_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S86_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S89_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S92_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S93_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S96_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S99_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S102_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S103_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S106_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S109_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S112_Conv2d_32x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S115_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S118_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S121_Conv2d_32x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S122_MatAdd_32x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S125_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S128_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S131_Conv2d_32x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S132_MatAdd_32x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S135_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S138_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S141_Conv2d_56x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S144_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S147_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S150_Conv2d_56x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S151_MatAdd_56x3x3(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S154_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S157_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S160_Conv2d_56x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S161_MatAdd_56x3x3(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S164_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S167_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S170_Conv2d_112x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S173_Conv2d_1280x112x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S176_Conv2d_1280x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S179_Conv2d_32x1280x1x1_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S180_Op_MEAN_0_72(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos);
extern void S183_Linear_2x32(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos);
extern void S184_SoftMax(
		signed char * __restrict__ In,
		short int * __restrict__ Out,
		signed char * __restrict__ Infos);
extern int classificationCNN_Construct();
extern int classificationCNN_Destruct();
extern int classificationCNN_Memory(int Which);
extern int classificationCNN(
		signed char * __restrict__ Input_1,
		signed short * __restrict__ Output_1);
#endif
