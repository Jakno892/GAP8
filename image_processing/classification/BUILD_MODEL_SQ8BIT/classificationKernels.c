#include "classificationKernels.h"
L1_CL_MEM AT_L1_POINTER classification_L1_Memory;
L2_MEM AT_L2_POINTER classification_L2_Memory;
L2_MEM AT_L2_POINTER classification_L2_Memory_Dyn;
AT_HYPERRAM_POINTER classification_L3_Memory;
static AT_HYPERRAM_T HyperRam;
static AT_HYPERFLASH_FS_T HyperFlash;
void S3_Conv2d_1x1x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46396 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Total=0, T1Ind_Last, T1Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 5][Tile0 Dim: 1]
	Ker Arg: In2, Tiled Space: Tile1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 79056 [Tile1, 5:[1x18544, 3:1x18544, 1x4880], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 5:[1x18544, 3:1x18544, 1x4880], 1]
		Tile0: [0, 18544, 18544], Tile1: [18544, 18544, 18544], Tile2; [37088, 18544, 18544]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [Tile0, 1:[1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 1]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 1 [Tile1, 5:[1x1, 3:1x1, 1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 5:[1x1, 3:1x1, 1x1], 1]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4 [Tile0, 1:[1x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 4]
		Tile0: [0, 4, 4], Tile1: [0, 4, 4], Tile2; [0, 4, 4]
	Ker Arg: Out, Tiled Space: Tile1
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 5 physical tiles
			Total Size: 19764 [Tile1, 5:[1x4636, 3:1x4636, 1x1220], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 5:[1x4636, 3:1x4636, 1x1220], 1]
		Tile0: [0, 4636, 4636], Tile1: [4636, 4636, 4636], Tile2; [9272, 4636, 4636]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [Tile0, 1:[1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 1]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1 [Tile0, 1:[1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 1]
		Tile0: [0, 1, 1], Tile1: [0, 1, 1], Tile2; [0, 1, 1]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 5 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 5:[1x1, 3:1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 5:[1x1, 3:1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+46380);
	KerArg0->W_In1 = (unsigned short int) (1);
	KerArg0->H_In1 = (unsigned short int) (1);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+46384);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+46388);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+46392);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Sx = (unsigned char) (2);
	KerArg0->Sy = (unsigned char) (2);
	KerArg0->W = (unsigned short int) (244);
	KerArg0->H = (unsigned short int) (324);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+46364);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4+0), 18544, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46380), 1, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46384), 4, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=4636;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46388), 1, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46392), 1, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46364), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T1Ind=0; T1Ind<5; T1Ind++, T1Ind_Total++) { /* Iteration on Tile1 */
		int T1Ind_Last = (T1Ind==4), T1Ind_NextLast = ((T1Ind+1)==4);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T1Ind_Last)) {
			_N_In2 = _N_In2 + (18544); _SN_In2 = ((T1Ind_NextLast)?4880:18544); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4+18544*((T1Ind_Total+1)%2)),
					_SN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+4+18544*((T1Ind_Total)%2));
			KerArg0->W_In2 = (unsigned short int) ((T1Ind_Last)?4880:18544);
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+37092+4636*((T1Ind_Total)%2));
			KerArg0->W_Out = (unsigned short int) ((T1Ind_Last)?1220:4636);
			KerArg0->OutFirstCol = (unsigned short int) ((0)*1);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46364))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulSxSyB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulSxSyB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37092+4636*((T1Ind_Total)%2)),
				_SC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T1Ind_Last)) {
			_C_Out = _C_Out + (4636); _SC_Out = ((T1Ind_NextLast)?1220:4636); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S6_Conv2d_3x1x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46696 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 4]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 19764 [Tile0, 4:[1x5184, 2:1x5184, 1x4212], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[1x5184, 2:1x5184, 1x4212], 1]
		Tile0: [0, 5184, 5184], Tile1: [5184, 5184, 5184], Tile2; [10368, 5184, 5184]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 3 [Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 19764 [Tile0, 4:[1x5184, 2:1x5184, 1x4212], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[1x5184, 2:1x5184, 1x4212], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 12 [Tile0, 4:[3x1, 2:3x1, 3x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[3x1, 2:3x1, 3x1], 4]
		Tile0: [0, 12, 12], Tile1: [0, 12, 12], Tile2; [0, 12, 12]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 59292 [Tile0, 4:[3x5184, 2:3x5184, 3x4212], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[3x5184, 2:3x5184, 3x4212], 1]
		Tile0: [0, 15552, 5184], Tile1: [5184, 15552, 5184], Tile2; [10368, 15552, 5184]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 3 [Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 3 [Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[3x1, 2:3x1, 3x1], 1]
		Tile0: [0, 3, 3], Tile1: [0, 3, 3], Tile2; [0, 3, 3]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 4:[1x1, 2:1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[1x1, 2:1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+10372);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (1);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (1);
	KerArg1->H_In1 = (unsigned short int) (3);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+10372);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+15556);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+46672);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+46676);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+46680);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4+0), 5184, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15556), 12, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=15552; _LC_Out=5184;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46672), 3, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46676), 3, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46680), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<4; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==3), T0Ind_NextLast = ((T0Ind+1)==3);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T0Ind_Last)) {
			_N_In2 = _N_In2 + (5184); _SN_In2 = ((T0Ind_NextLast)?4212:5184); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4+5184*((T0Ind_Total+1)%2)),
					_SN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+4+5184*((T0Ind_Total)%2));
		KerArg0->W = (unsigned short int) ((T0Ind_Last)?4212:5184);
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->W_In2 = (unsigned short int) ((T0Ind_Last)?4212:5184);
		KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+15568+15552*((T0Ind_Total)%2));
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46680))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_SF_SQ8, KerArg1);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15568+15552*((T0Ind_Total)%2)),
				_SC_Out, 19764, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (5184); _LC_Out = ((T0Ind_NextLast)?4212:5184); _SC_Out = (3*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S7_Op_RESIZE_BILINEAR_0_3(
		signed char * In,
		signed char * Out)

{
	/* Shared L1: 41360 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	KerResizeBilinearSigned_ArgT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _N_In, _Off_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: 3][Tile0 Dim: 2]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 6 logical tiles, 6 physical tiles
			Total Size: 27648 [D0, [2 x 9216, 9216]][Tile0, 2:[96x68, 96x28], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 9216, 9216]][Tile0, 2:[96x68, 96x28], 1]
		Tile0: [0, 6528, 6528], Tile1: [6528, 2688, 2688], Tile2; [9216, 6528, 6528]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 6 logical tiles, 6 physical tiles
			Total Size: 59292 [D0, [2 x 19764, 19764]][Tile0, 2:[122x116, 122x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 19764, 19764]][Tile0, 2:[122x116, 122x48], 1]
		Tile0: [0, 14152, 14152], Tile1: [13908, 5856, 5856], Tile2; [19764, 14152, 14152]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Win = (unsigned int) (122);
	KerArg0->Hin = (unsigned int) (162);
	KerArg0->Wout = (unsigned int) (96);
	KerArg0->Hout = (unsigned int) (96);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=6528;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 14152, 0, DmaR_Evt1);
	_N_In=0;
	/*============================= End Read Tiles Prolog ===============================*/
	for (D0Ind=0; D0Ind<3; D0Ind++) { /* Iteration on D0 */
		int D0Ind_Last = (D0Ind==2), D0Ind_NextLast = ((D0Ind+1)==2);
		for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + 0; _Off_In = (((109909*((T0Ind)+1)*68)>>16)*122); _SN_In = ((1)?5856:14152); 
			} else if (!(D0Ind_Last)) {
				_N_In = _N_In + (19764); _Off_In = 0; _SN_In = (14152); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In+_Off_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+14152*((T0Ind_Total+1)%2)),
						_SN_In, 0, DmaR_Evt1);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0+14152*((T0Ind_Total)%2));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+28304+6528*((T0Ind_Total)%2));
			KerArg0->HTileOut = (unsigned int) (T0Ind_Last?28:68);
			KerArg0->FirstLineIndex = (unsigned int) ((109909*(T0Ind)*68)>>16);
			AT_FORK(gap_ncore(), (void *) KerResizeBilinearSigned, (void *) KerArg0);
			__CALL(KerResizeBilinearSigned, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+28304+6528*((T0Ind_Total)%2)),
					_SC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (6528); _SC_Out = ((1)?2688:6528); 
			} else if (!(D0Ind_Last)) {
				_C_Out = _C_Out + (9216)+(-6528); _SC_Out = (6528); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S10_Conv2d_16x3x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 41440 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 7][D0 Dim: Init: 3, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 36864 [D1, [0 x 36864, 36864]][Tile0, 7:[48x7, 5:48x7, 48x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 36864, 36864]][Tile0, 7:[48x7, 5:48x7, 48x6], 1]
		Tile0: [0, 5376, 336], Tile1: [336, 5376, 336], Tile2; [672, 5376, 336]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 432 [D1, [0 x 432, 432]][D0, [0 x 432, 432]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 432, 432]][D0, [0 x 432, 432]]
		Tile0: [0, 432, 432], Tile1: [0, 432, 432], Tile2; [0, 432, 432]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 27648 [D0, [0 x 27648, 27648]][Tile0, 7:[96x15, 5:96x15, 96x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[96x14], 1][D0, [0 x 27648, 27648]]
		Tile0: [0, 4320, 1440], Tile1: [1344, 4320, 1440], Tile2; [2688, 4320, 1440]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 147456 [D1, [0 x 147456, 147456]][Tile0, 7:[48x7, 5:48x7, 48x6], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 147456, 147456]][Tile0, 7:[48x7, 5:48x7, 48x6], 4]
		Tile0: [0, 21504, 1344], Tile1: [0, 21504, 1344], Tile2; [0, 21504, 1344]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 7:[13x1, 5:13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[13x1, 5:13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+19920);
	KerArg0->W = (unsigned short int) (48);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+8640);
	KerArg1->W = (unsigned short int) (96);
	KerArg1->UsedW = (unsigned short int) (96);
	KerArg1->InFeatures = (unsigned short int) (3);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (3);
	KerArg1->Filter = (signed char * __restrict__) (classification_L1_Memory+8736);
	KerArg1->Out = (int * __restrict__) (classification_L1_Memory+19920);
	KerArg2->In = (int *__restrict__) (classification_L1_Memory+19920);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (48);
	KerArg2->Scale = (unsigned char *__restrict__) (classification_L1_Memory+8704);
	KerArg2->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+8720);
	KerArg2->Infos = (signed char *__restrict__) (classification_L1_Memory+41424);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=5376; _LC_Out=336;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8640), 64, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8704), 16, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8720), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8736), 432, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 4320, 9216, 1440, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41424), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<7; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==6), T0Ind_NextLast = ((T0Ind+1)==6);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (1344); _LN_In = ((T0Ind_NextLast)?1152:1440); _SN_In = (3*_LN_In); 
			} else if (!(1)) {
				_N_In = _N_In + (-8064); _LN_In = (1440); _SN_In = (3*_LN_In); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+4320*((T0Ind_Total+1)%2)),
						_SN_In, 9216, _LN_In, 0, DmaR_Evt5);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->H = (unsigned short int) (T0Ind_Last?6:7);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+41424))[8]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (classification_L1_Memory+0+4320*((T0Ind_Total)%2));
				KerArg1->H = (unsigned short int) (((T0Ind_Last)?12:15)-0*(T0Ind==0));
				KerArg1->UsedH = (unsigned short int) (((T0Ind_Last)?12:15)-0*(T0Ind==0));
				KerArg1->Pad = (v4s) ((v4s){0,1,0*(T0Ind==0),1*(T0Ind_Last)});
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride2_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride2_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			KerArg2->Out = (void *__restrict__) (classification_L1_Memory+9168+5376*((T0Ind_Total)%2));
			KerArg2->H = (unsigned short int) (T0Ind_Last?6:7);
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9168+5376*((T0Ind_Total)%2)),
					_SC_Out, 2304, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (336); _LC_Out = ((T0Ind_NextLast)?288:336); _SC_Out = (16*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S13_Conv2d_16x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46336 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 2][Tile0 Dim: 2]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 36864 [D0, [1 x 18432, 18432]][Tile0, 2:[48x30, 48x20], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 18432, 18432]][Tile0, 2:[48x30, 48x20], 1]
		Tile0: [0, 11520, 1440], Tile1: [1344, 7680, 960], Tile2; [18432, 11520, 1440]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 64 [D0, [1 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 32, 32]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 16 [D0, [1 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 8, 8]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 16 [D0, [1 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 8, 8]]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [1 x 72, 72]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 72, 72]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 36864 [D0, [1 x 18432, 18432]][Tile0, 2:[48x29, 48x19], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [1 x 18432, 18432]][Tile0, 2:[48x29, 48x19], 1]
		Tile0: [0, 11136, 1392], Tile1: [1392, 7296, 912], Tile2; [18432, 11136, 1392]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 2:[13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (48);
	KerArg0->UsedW = (unsigned short int) (48);
	KerArg0->InFeatures = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->TotalInFeatures = (unsigned short int) (16);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+46320);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 11520, 2304, 1440, 0, DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23808), 64, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23872), 16, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23888), 16, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23904), 144, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=11136; _LC_Out=1392;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46320), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (D0Ind=0; D0Ind<2; D0Ind++) { /* Iteration on D0 */
		int D0Ind_Last = (D0Ind==1), D0Ind_NextLast = ((D0Ind+1)==1);
		for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (1392-(48*(T0Ind==0))); _LN_In = ((1)?960:1488); _SN_In = (8*_LN_In); 
			} else if (!(D0Ind_Last)) {
				_N_In = _N_In + (18432)+(-1344); _LN_In = (1440); _SN_In = (8*_LN_In); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+11904*((T0Ind_Total+1)%2)),
						_SN_In, 2304, _LN_In, 0, DmaR_Evt1);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0+11904*((T0Ind_Total)%2));
			KerArg0->H = (unsigned short int) (((T0Ind_Last)?20:31)-1*(T0Ind==0));
			KerArg0->UsedH = (unsigned short int) (((T0Ind_Last)?20:31)-1*(T0Ind==0));
			KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+23904+((D0Ind)*72));
			KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+23808+((D0Ind)*32));
			KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+23872+((D0Ind)*8));
			KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+23888+((D0Ind)*8));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+24048+11136*((T0Ind_Total)%2));
			KerArg0->Pad = (v4s) ((v4s){1,1,1*(T0Ind==0),1*(T0Ind_Last)});
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46320))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24048+11136*((T0Ind_Total)%2)),
					_SC_Out, 2304, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (1392); _LC_Out = ((1)?912:1392); _SC_Out = (8*_LC_Out); 
			} else if (!(D0Ind_Last)) {
				_C_Out = _C_Out + (18432)+(-1392); _LC_Out = (1392); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S16_Conv2d_8x16x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46272 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 4]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 36864 [Tile0, 4:[16x720, 2:16x720, 16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[16x720, 2:16x720, 16x144], 1]
		Tile0: [0, 11520, 720], Tile1: [720, 11520, 720], Tile2; [1440, 11520, 720]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 4:[8x16, 2:8x16, 8x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[8x16, 2:8x16, 8x16], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 36864 [Tile0, 4:[16x720, 2:16x720, 16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[16x720, 2:16x720, 16x144], 1]
		Tile0: [0, 11520, 720], Tile1: [0, 11520, 720], Tile2; [0, 11520, 720]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 4:[8x1, 2:8x1, 8x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[8x1, 2:8x1, 8x1], 4]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 4 physical tiles
			Total Size: 18432 [Tile0, 4:[8x720, 2:8x720, 8x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[8x720, 2:8x720, 8x144], 1]
		Tile0: [0, 5760, 720], Tile1: [720, 5760, 720], Tile2; [1440, 5760, 720]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 4:[8x1, 2:8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[8x1, 2:8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 4:[8x1, 2:8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[8x1, 2:8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 4 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 4:[1x1, 2:1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 4:[1x1, 2:1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+23168);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (16);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (16);
	KerArg1->H_In1 = (unsigned short int) (8);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+23168);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+34688);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+46240);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+46248);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+46256);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+128+0), 11520, 2304, 720, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 128, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+34688), 32, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=5760; _LC_Out=720;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46240), 8, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46248), 8, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46256), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<4; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==3), T0Ind_NextLast = ((T0Ind+1)==3);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T0Ind_Last)) {
			_N_In2 = _N_In2 + (720); _LN_In2 = ((T0Ind_NextLast)?144:720); _SN_In2 = (16*_LN_In2); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+128+11520*((T0Ind_Total+1)%2)),
					_SN_In2, 2304, _LN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+128+11520*((T0Ind_Total)%2));
		KerArg0->W = (unsigned short int) ((T0Ind_Last)?144:720);
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->W_In2 = (unsigned short int) ((T0Ind_Last)?144:720);
		KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+34720+5760*((T0Ind_Total)%2));
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46256))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_SF_SQ8, KerArg1);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+34720+5760*((T0Ind_Total)%2)),
				_SC_Out, 2304, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (720); _LC_Out = ((T0Ind_NextLast)?144:720); _SC_Out = (8*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S19_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 45808 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 7]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 18432 [Tile0, 7:[8x376, 5:8x376, 8x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[8x376, 5:8x376, 8x48], 1]
		Tile0: [0, 3008, 376], Tile1: [376, 3008, 376], Tile2; [752, 3008, 376]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 7:[48x8, 5:48x8, 48x8], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[48x8, 5:48x8, 48x8], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 18432 [Tile0, 7:[8x376, 5:8x376, 8x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[8x376, 5:8x376, 8x48], 1]
		Tile0: [0, 3008, 376], Tile1: [0, 3008, 376], Tile2; [0, 3008, 376]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 192 [Tile0, 7:[48x1, 5:48x1, 48x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[48x1, 5:48x1, 48x1], 4]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 7 physical tiles
			Total Size: 110592 [Tile0, 7:[48x376, 5:48x376, 48x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[48x376, 5:48x376, 48x48], 1]
		Tile0: [0, 18048, 376], Tile1: [376, 18048, 376], Tile2; [752, 18048, 376]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 7:[48x1, 5:48x1, 48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[48x1, 5:48x1, 48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 7:[48x1, 5:48x1, 48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[48x1, 5:48x1, 48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 7 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 7:[1x1, 5:1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 7:[1x1, 5:1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+6400);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (8);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (8);
	KerArg1->H_In1 = (unsigned short int) (48);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+6400);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+9408);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+45696);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+45744);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+45792);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+0), 3008, 2304, 376, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9408), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=18048; _LC_Out=376;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+45696), 48, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+45744), 48, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+45792), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<7; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==6), T0Ind_NextLast = ((T0Ind+1)==6);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T0Ind_Last)) {
			_N_In2 = _N_In2 + (376); _LN_In2 = ((T0Ind_NextLast)?48:376); _SN_In2 = (8*_LN_In2); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+3008*((T0Ind_Total+1)%2)),
					_SN_In2, 2304, _LN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+384+3008*((T0Ind_Total)%2));
		KerArg0->W = (unsigned short int) ((T0Ind_Last)?48:376);
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->W_In2 = (unsigned short int) ((T0Ind_Last)?48:376);
		KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+9600+18048*((T0Ind_Total)%2));
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+45792))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600+18048*((T0Ind_Total)%2)),
				_SC_Out, 2304, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (376); _LC_Out = ((T0Ind_NextLast)?48:376); _SC_Out = (48*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S22_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 45664 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 48, Tiled: 6][Tile0 Dim: 2]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 27648 [D0, [5 x 4608, 4608]][Tile0, 2:[24x23, 24x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 4608, 4608]][Tile0, 2:[24x23, 24x1], 1]
		Tile0: [0, 4416, 552], Tile1: [552, 192, 24], Tile2; [4608, 4416, 552]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 6 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [5 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 32, 32]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 6 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [5 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 8, 8]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 6 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [5 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 8, 8]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 6 logical tiles, 1 physical tiles
			Total Size: 432 [D0, [5 x 72, 72]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 72, 72]]
		Tile0: [0, 432, 432], Tile1: [0, 432, 432], Tile2; [0, 432, 432]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 12 logical tiles, 12 physical tiles
			Total Size: 110592 [D0, [5 x 18432, 18432]][Tile0, 2:[48x47, 48x2], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [5 x 18432, 18432]][Tile0, 2:[48x47, 48x2], 1]
		Tile0: [0, 18048, 2256], Tile1: [2208, 768, 96], Tile2; [18432, 18048, 2256]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 2:[13x1, 13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[13x1, 13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (48);
	KerArg0->UsedW = (unsigned short int) (48);
	KerArg0->InFeatures = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg0->TotalInFeatures = (unsigned short int) (48);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+45648);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=4416; _LC_Out=552;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+36096), 192, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+36288), 48, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+36336), 48, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+36384), 432, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 18048, 2304, 2256, 0, DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+45648), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (D0Ind=0; D0Ind<6; D0Ind++) { /* Iteration on D0 */
		int D0Ind_Last = (D0Ind==5), D0Ind_NextLast = ((D0Ind+1)==5);
		for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (2208); _LN_In = ((1)?96:2256); _SN_In = (8*_LN_In); 
			} else if (!(D0Ind_Last)) {
				_N_In = _N_In + (18432)+(-2208); _LN_In = (2256); _SN_In = (8*_LN_In); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+18048*((T0Ind_Total+1)%2)),
						_SN_In, 2304, _LN_In, 0, DmaR_Evt5);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0+18048*((T0Ind_Total)%2));
			KerArg0->H = (unsigned short int) (((T0Ind_Last)?2:47)-0*(T0Ind==0));
			KerArg0->UsedH = (unsigned short int) (((T0Ind_Last)?2:47)-0*(T0Ind==0));
			KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+36384+((D0Ind)*72));
			KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+36096+((D0Ind)*32));
			KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+36288+((D0Ind)*8));
			KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+36336+((D0Ind)*8));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+36816+4416*((T0Ind_Total)%2));
			KerArg0->Pad = (v4s) ((v4s){0,1,0*(T0Ind==0),1*(T0Ind_Last)});
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+45648))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, KerArg0);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+36816+4416*((T0Ind_Total)%2)),
					_SC_Out, 576, _LC_Out, 1, DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (552); _LC_Out = ((1)?24:552); _SC_Out = (8*_LC_Out); 
			} else if (!(D0Ind_Last)) {
				_C_Out = _C_Out + (4608)+(-552); _LC_Out = (552); _SC_Out = (8*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S25_Conv2d_8x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46528 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 2]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 27648 [Tile0, 2:[48x288, 48x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[48x288, 48x288], 1]
		Tile0: [0, 13824, 288], Tile1: [288, 13824, 288], Tile2; [0, 13824, 288]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 2:[8x48, 8x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x48, 8x48], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 27648 [Tile0, 2:[48x288, 48x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[48x288, 48x288], 1]
		Tile0: [0, 13824, 288], Tile1: [0, 13824, 288], Tile2; [0, 13824, 288]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 2:[8x1, 8x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 4]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 4608 [Tile0, 2:[8x288, 8x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x288, 8x288], 1]
		Tile0: [0, 2304, 288], Tile1: [288, 2304, 288], Tile2; [0, 2304, 288]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 2:[8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 2:[8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 2:[1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+28032);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (288);
	KerArg0->H = (unsigned short int) (48);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (48);
	KerArg1->H_In1 = (unsigned short int) (8);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+28032);
	KerArg1->W_In2 = (unsigned short int) (288);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+41856);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+46496);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+46504);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+46512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+0), 13824, 576, 288, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41856), 32, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=2304; _LC_Out=288;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46496), 8, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46504), 8, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46512), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T0Ind_Last)) {
			_N_In2 = _N_In2 + (288); _LN_In2 = (288); _SN_In2 = (48*_LN_In2); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+13824*((T0Ind_Total+1)%2)),
					_SN_In2, 576, _LN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+384+13824*((T0Ind_Total)%2));
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+41888+2304*((T0Ind_Total)%2));
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46512))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_SF_SQ8, KerArg1);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41888+2304*((T0Ind_Total)%2)),
				_SC_Out, 576, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (288); _LC_Out = (288); _SC_Out = (8*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S28_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 37552 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[8x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[8x576], 1]
		Tile0: [0, 4608, 576], Tile1: [0, 4608, 576], Tile2; [0, 4608, 576]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[8x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[8x576], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 1:[48x8], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x8], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile0, 1:[48x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 4]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 27648 [Tile0, 1:[48x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x576], 1]
		Tile0: [0, 27648, 27648], Tile1: [0, 27648, 27648], Tile2; [0, 27648, 27648]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 1:[48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 1:[48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+384);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+4992);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (576);
	KerArg0->H = (unsigned short int) (8);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (8);
	KerArg1->H_In1 = (unsigned short int) (48);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+4992);
	KerArg1->W_In2 = (unsigned short int) (576);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+9600);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+37440);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+37488);
	KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+9792);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+37536);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384), 4608, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37440), 48, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37488), 48, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37536), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+37536))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9792), 27648, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S31_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 37600 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 48, Tiled: 3][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 27648 [D0, [2 x 9216, 9216]][Tile0, 1:[24x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 9216, 9216]][Tile0, 1:[24x24], 1]
		Tile0: [0, 9216, 576], Tile1: [9216, 9216, 576], Tile2; [18432, 9216, 576]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [2 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 64, 64]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [2 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 16, 16]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [2 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 16, 16]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 432 [D0, [2 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 144, 144]]
		Tile0: [0, 432, 432], Tile1: [0, 432, 432], Tile2; [0, 432, 432]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 27648 [D0, [2 x 9216, 9216]][Tile0, 1:[24x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [2 x 9216, 9216]][Tile0, 1:[24x24], 1]
		Tile0: [0, 9216, 576], Tile1: [9216, 9216, 576], Tile2; [18432, 9216, 576]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W = (unsigned short int) (24);
	KerArg0->UsedW = (unsigned short int) (24);
	KerArg0->InFeatures = (unsigned short int) (16);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg0->TotalInFeatures = (unsigned short int) (48);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+37584);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 9216, 576, 576, 0, DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18432), 192, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18624), 48, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18672), 48, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18720), 432, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	_C_Out=0; _SC_Out=9216; _LC_Out=576;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37584), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (D0Ind=0; D0Ind<3; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
		int D0Ind_Last = (D0Ind==2), D0Ind_NextLast = ((D0Ind+1)==2);
		/*================================= Prepare Tiles ===================================*/
		_SN_In = 0;
		if (!(D0Ind_Last)) {
			_N_In = _N_In + (9216); _LN_In = (576); _SN_In = (16*_LN_In); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
		if (_SN_In) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+9216*((D0Ind_Total+1)%2)),
					_SN_In, 576, _LN_In, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0+9216*((D0Ind_Total)%2));
			KerArg0->H = (unsigned short int) (26-1*(1)-1*(1));
			KerArg0->UsedH = (unsigned short int) (26-1*(1)-1*(1));
			KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+18720+((D0Ind)*144));
			KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+18432+((D0Ind)*64));
			KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+18624+((D0Ind)*16));
			KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+18672+((D0Ind)*16));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+19152+9216*((D0Ind_Total)%2));
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+37584))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19152+9216*((D0Ind_Total)%2)),
				_SC_Out, 576, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(D0Ind_Last)) {
			_C_Out = _C_Out + (9216); _LC_Out = (576); _SC_Out = (16*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S34_Conv2d_8x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 46528 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In2;
	unsigned int _SN_In2;
	unsigned int _LN_In2;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 2]
	Ker Arg: In2, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 27648 [Tile0, 2:[48x288, 48x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[48x288, 48x288], 1]
		Tile0: [0, 13824, 288], Tile1: [288, 13824, 288], Tile2; [0, 13824, 288]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 2:[8x48, 8x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x48, 8x48], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 27648 [Tile0, 2:[48x288, 48x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[48x288, 48x288], 1]
		Tile0: [0, 13824, 288], Tile1: [0, 13824, 288], Tile2; [0, 13824, 288]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 2:[8x1, 8x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 4]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 4608 [Tile0, 2:[8x288, 8x288], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x288, 8x288], 1]
		Tile0: [0, 2304, 288], Tile1: [288, 2304, 288], Tile2; [0, 2304, 288]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 2:[8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 8 [Tile0, 2:[8x1, 8x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[8x1, 8x1], 1]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 2:[1x1, 1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[1x1, 1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+28032);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (288);
	KerArg0->H = (unsigned short int) (48);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (48);
	KerArg1->H_In1 = (unsigned short int) (8);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+28032);
	KerArg1->W_In2 = (unsigned short int) (288);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+41856);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+46496);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+46504);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+46512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+0), 13824, 576, 288, 0, DmaR_Evt1);
	_N_In2=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41856), 32, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=2304; _LC_Out=288;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46496), 8, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46504), 8, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+46512), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
		int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
		/*================================= Prepare Tiles ===================================*/
		_SN_In2 = 0;
		if (!(T0Ind_Last)) {
			_N_In2 = _N_In2 + (288); _LN_In2 = (288); _SN_In2 = (48*_LN_In2); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
		if (_SN_In2) {
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In2+_N_In2), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384+13824*((T0Ind_Total+1)%2)),
					_SN_In2, 576, _LN_In2, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+384+13824*((T0Ind_Total)%2));
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+41888+2304*((T0Ind_Total)%2));
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+46512))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_SF_SQ8, KerArg1);
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41888+2304*((T0Ind_Total)%2)),
				_SC_Out, 576, _LC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;_LP_Out = _LC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T0Ind_Last)) {
			_C_Out = _C_Out + (288); _LC_Out = (288); _SC_Out = (8*_LC_Out); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S35_MatAdd_8x24x24(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 13840 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[1x4608], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x4608], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[1x4608], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x4608], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[1x4608], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x4608], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+4608);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+9216);
	KerArg0->Feat = (unsigned short int) (4608);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+13824);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 4608, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4608), 4608, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9216), 4608, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S38_Conv2d_48x8x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 37552 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[8x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[8x576], 1]
		Tile0: [0, 4608, 576], Tile1: [0, 4608, 576], Tile2; [0, 4608, 576]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[8x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[8x576], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 1:[48x8], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x8], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile0, 1:[48x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 4]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 27648 [Tile0, 1:[48x576], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x576], 1]
		Tile0: [0, 27648, 27648], Tile1: [0, 27648, 27648], Tile2; [0, 27648, 27648]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 1:[48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [Tile0, 1:[48x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x1], 1]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+384);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+4992);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (576);
	KerArg0->H = (unsigned short int) (8);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (8);
	KerArg1->H_In1 = (unsigned short int) (48);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+4992);
	KerArg1->W_In2 = (unsigned short int) (576);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+9600);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+37440);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+37488);
	KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+9792);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+37536);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384), 4608, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37440), 48, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37488), 48, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37536), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+37536))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_ReLU_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9792), 27648, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S41_Conv2d_48x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 35296 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 48, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 27648 [D0, [0 x 27648, 27648]][Tile0, 1:[24x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 27648, 27648]][Tile0, 1:[24x24], 1]
		Tile0: [0, 27648, 27648], Tile1: [0, 27648, 27648], Tile2; [0, 27648, 27648]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [0 x 48, 48]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 48, 48]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 48 [D0, [0 x 48, 48]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 48, 48]]
		Tile0: [0, 48, 48], Tile1: [0, 48, 48], Tile2; [0, 48, 48]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 432 [D0, [0 x 432, 432]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 432, 432]]
		Tile0: [0, 432, 432], Tile1: [0, 432, 432], Tile2; [0, 432, 432]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[12x12], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (24);
	KerArg0->UsedW = (unsigned short int) (24);
	KerArg0->H = (unsigned short int) (24);
	KerArg0->InFeatures = (unsigned short int) (48);
	KerArg0->OutFeatures = (unsigned short int) (48);
	KerArg0->TotalInFeatures = (unsigned short int) (48);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+27936);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+27648);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+27840);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+27888);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+35280);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+28368);
	KerArg0->Pad = (v4s) ((v4s){0,1,0,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 27648, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+27648), 192, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+27840), 48, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+27888), 48, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+27936), 432, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+35280), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (24);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+35280))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+28368), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S44_Conv2d_16x48x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 17008 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatTranspose_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerMatMul_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: TransIn2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile0, 1:[48x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x144], 1]
		Tile0: [0, 6912, 144], Tile1: [0, 6912, 144], Tile2; [0, 6912, 144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile0, 1:[48x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[48x144], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile0, 1:[16x48], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x48], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[16x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 4]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[16x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x1], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+768);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+7680);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->W = (unsigned short int) (144);
	KerArg0->H = (unsigned short int) (48);
	KerArg1->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg1->W_In1 = (unsigned short int) (48);
	KerArg1->H_In1 = (unsigned short int) (16);
	KerArg1->In2 = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg1->W_In2 = (unsigned short int) (144);
	KerArg1->Bias = (void * __restrict__) (classification_L1_Memory+14592);
	KerArg1->Scale = (unsigned char * __restrict__) (classification_L1_Memory+16960);
	KerArg1->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+16976);
	KerArg1->Out = (signed char * __restrict__) (classification_L1_Memory+14656);
	KerArg1->Infos = (signed char *__restrict__) (classification_L1_Memory+16992);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+768), 6912, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 768, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14592), 64, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16960), 16, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16976), 16, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16992), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_Transpose_fps, (void *) KerArg0);
		__CALL(CNN_Transpose_fps, KerArg0);
		KerArg1->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16992))[8]);
		AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SF_SQ8, (void *) KerArg1);
		__CALL(KerParMatMulB32_SF_SQ8, KerArg1);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14656), 2304, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S47_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18320 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[64x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[64x1], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1536 [Tile0, 1:[16x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x96], 1]
		Tile0: [0, 1536, 1536], Tile1: [0, 1536, 1536], Tile2; [0, 1536, 1536]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile1, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 1:[1x96], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 4]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [Tile1, 1:[96x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x144], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+16208);
	KerArg0->W_In1 = (unsigned short int) (16);
	KerArg0->H_In1 = (unsigned short int) (96);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+64);
	KerArg0->W_In2 = (unsigned short int) (144);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+17744);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+18128);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+18224);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+2368);
	KerArg0->W_Out = (unsigned short int) (144);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+16192);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16208), 1536, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+64), 2304, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+17744), 384, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18128), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18224), 96, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16192), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*96);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16192))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2368), 13824, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S50_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 96, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [D0, [0 x 384, 384]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 384, 384]]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [D0, [0 x 864, 864]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 864, 864]]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (12);
	KerArg0->UsedW = (unsigned short int) (12);
	KerArg0->H = (unsigned short int) (12);
	KerArg0->InFeatures = (unsigned short int) (96);
	KerArg0->OutFeatures = (unsigned short int) (96);
	KerArg0->TotalInFeatures = (unsigned short int) (96);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+14400);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+14304);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+29088);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+15264);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 13824, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14208), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14304), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14400), 864, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+29088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (12);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+29088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15264), 13824, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S53_Conv2d_16x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18160 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile1, 1:[384x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[384x1], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1536 [Tile0, 1:[96x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x16], 1]
		Tile0: [0, 1536, 1536], Tile1: [0, 1536, 1536], Tile2; [0, 1536, 1536]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [Tile1, 1:[96x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x144], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[1x16], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 4]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile1, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+16528);
	KerArg0->W_In1 = (unsigned short int) (96);
	KerArg0->H_In1 = (unsigned short int) (16);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+384);
	KerArg0->W_In2 = (unsigned short int) (144);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18064);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+18128);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+18144);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->W_Out = (unsigned short int) (144);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+16512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16528), 1536, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384), 13824, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18064), 64, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18128), 16, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18144), 16, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16512), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*16);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16512))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14208), 2304, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S54_MatAdd_16x12x12(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 6928 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+2304);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+4608);
	KerArg0->Feat = (unsigned short int) (2304);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6912);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 2304, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2304), 2304, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4608), 2304, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S57_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18320 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[64x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[64x1], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1536 [Tile0, 1:[16x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x96], 1]
		Tile0: [0, 1536, 1536], Tile1: [0, 1536, 1536], Tile2; [0, 1536, 1536]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile1, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 1:[1x96], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 4]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [Tile1, 1:[96x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x144], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+16208);
	KerArg0->W_In1 = (unsigned short int) (16);
	KerArg0->H_In1 = (unsigned short int) (96);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+64);
	KerArg0->W_In2 = (unsigned short int) (144);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+17744);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+18128);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+18224);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+2368);
	KerArg0->W_Out = (unsigned short int) (144);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+16192);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16208), 1536, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+64), 2304, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+17744), 384, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18128), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18224), 96, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16192), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*96);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16192))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2368), 13824, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S60_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 96, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [D0, [0 x 384, 384]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 384, 384]]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [D0, [0 x 864, 864]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 864, 864]]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (12);
	KerArg0->UsedW = (unsigned short int) (12);
	KerArg0->H = (unsigned short int) (12);
	KerArg0->InFeatures = (unsigned short int) (96);
	KerArg0->OutFeatures = (unsigned short int) (96);
	KerArg0->TotalInFeatures = (unsigned short int) (96);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+14400);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+14304);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+29088);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+15264);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 13824, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14208), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14304), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14400), 864, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+29088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (12);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+29088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15264), 13824, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S63_Conv2d_16x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18160 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile1, 1:[384x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[384x1], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1536 [Tile0, 1:[96x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x16], 1]
		Tile0: [0, 1536, 1536], Tile1: [0, 1536, 1536], Tile2; [0, 1536, 1536]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [Tile1, 1:[96x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x144], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[1x16], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 4]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile1, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [Tile0, 1:[1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x16], 1]
		Tile0: [0, 16, 16], Tile1: [0, 16, 16], Tile2; [0, 16, 16]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+16528);
	KerArg0->W_In1 = (unsigned short int) (96);
	KerArg0->H_In1 = (unsigned short int) (16);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+384);
	KerArg0->W_In2 = (unsigned short int) (144);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18064);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+18128);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+18144);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->W_Out = (unsigned short int) (144);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+16512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16528), 1536, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384), 13824, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18064), 64, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18128), 16, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18144), 16, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16512), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*16);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16512))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14208), 2304, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S64_MatAdd_16x12x12(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 6928 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[1x2304], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x2304], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+2304);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+4608);
	KerArg0->Feat = (unsigned short int) (2304);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6912);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 2304, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2304), 2304, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4608), 2304, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S67_Conv2d_96x16x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18320 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile1, 1:[64x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[64x1], 1]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1536 [Tile0, 1:[16x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[16x96], 1]
		Tile0: [0, 1536, 1536], Tile1: [0, 1536, 1536], Tile2; [0, 1536, 1536]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile1, 1:[16x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[16x144], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile0, 1:[1x96], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 4]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [Tile1, 1:[96x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x144], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x96], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x96], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+16208);
	KerArg0->W_In1 = (unsigned short int) (16);
	KerArg0->H_In1 = (unsigned short int) (96);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+64);
	KerArg0->W_In2 = (unsigned short int) (144);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+17744);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+18128);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+18224);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+2368);
	KerArg0->W_Out = (unsigned short int) (144);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+16192);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16208), 1536, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+64), 2304, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+17744), 384, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18128), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18224), 96, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16192), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*96);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16192))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2368), 13824, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S70_Conv2d_96x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 18736 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 96, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13824 [D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 13824, 13824]][Tile0, 1:[12x12], 1]
		Tile0: [0, 13824, 13824], Tile1: [0, 13824, 13824], Tile2; [0, 13824, 13824]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [D0, [0 x 384, 384]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 384, 384]]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [D0, [0 x 96, 96]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 96, 96]]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [D0, [0 x 864, 864]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 864, 864]]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [D0, [0 x 3456, 3456]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3456, 3456]][Tile0, 1:[6x6], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (12);
	KerArg0->UsedW = (unsigned short int) (12);
	KerArg0->H = (unsigned short int) (12);
	KerArg0->InFeatures = (unsigned short int) (96);
	KerArg0->OutFeatures = (unsigned short int) (96);
	KerArg0->TotalInFeatures = (unsigned short int) (96);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+14400);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+14304);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+18720);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+15264);
	KerArg0->Pad = (v4s) ((v4s){0,1,0,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 13824, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 384, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14208), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14304), 96, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14400), 864, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18720), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (12);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+18720))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15264), 3456, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S73_Conv2d_24x96x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 7168 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 384 [Tile1, 1:[384x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[384x1], 1]
		Tile0: [0, 384, 384], Tile1: [0, 384, 384], Tile2; [0, 384, 384]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2304 [Tile0, 1:[96x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x24], 1]
		Tile0: [0, 2304, 2304], Tile1: [0, 2304, 2304], Tile2; [0, 2304, 2304]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile1, 1:[96x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[96x36], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 4]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile1, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+4720);
	KerArg0->W_In1 = (unsigned short int) (96);
	KerArg0->H_In1 = (unsigned short int) (24);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+384);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+7024);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+7120);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+7144);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+3840);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+4704);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4720), 2304, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+384), 3456, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7024), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7120), 24, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7144), 24, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4704), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*24);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+4704))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3840), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S76_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[96x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x1], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile1, 1:[24x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x144], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[1x144], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 4]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[36x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x144], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (24);
	KerArg0->H_In1 = (unsigned short int) (144);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+9600);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+3456);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+9216);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+9360);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+4032);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+9504);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+10464);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 576, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9216), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9360), 144, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10464), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+10464))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4032), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S79_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12544 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 144, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [D0, [0 x 576, 576]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 576, 576]]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1296 [D0, [0 x 1296, 1296]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1296, 1296]]
		Tile0: [0, 1296, 1296], Tile1: [0, 1296, 1296], Tile2; [0, 1296, 1296]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (144);
	KerArg0->OutFeatures = (unsigned short int) (144);
	KerArg0->TotalInFeatures = (unsigned short int) (144);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+6048);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+5184);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+5904);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+12528);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7344);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 5184, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5184), 576, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 144, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5904), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6048), 1296, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+12528), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+12528))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7344), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S82_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10240 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[576x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[576x1], 1]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile0, 1:[144x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[144x24], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[144x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[144x36], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 4]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile1, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+6640);
	KerArg0->W_In1 = (unsigned short int) (144);
	KerArg0->H_In1 = (unsigned short int) (24);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+576);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+10096);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+10192);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+10216);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6624);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6640), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+576), 5184, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10096), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10192), 24, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10216), 24, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6624), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*24);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+6624))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S83_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 2608 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+864);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+1728);
	KerArg0->Feat = (unsigned short int) (864);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+2592);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 864, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+864), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2592), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1728), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S86_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[96x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x1], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile1, 1:[24x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x144], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[1x144], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 4]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[36x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x144], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (24);
	KerArg0->H_In1 = (unsigned short int) (144);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+9600);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+3456);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+9216);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+9360);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+4032);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+9504);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+10464);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 576, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9216), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9360), 144, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10464), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+10464))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4032), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S89_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12544 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 144, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [D0, [0 x 576, 576]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 576, 576]]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1296 [D0, [0 x 1296, 1296]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1296, 1296]]
		Tile0: [0, 1296, 1296], Tile1: [0, 1296, 1296], Tile2; [0, 1296, 1296]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (144);
	KerArg0->OutFeatures = (unsigned short int) (144);
	KerArg0->TotalInFeatures = (unsigned short int) (144);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+6048);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+5184);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+5904);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+12528);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7344);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 5184, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5184), 576, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 144, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5904), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6048), 1296, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+12528), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+12528))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7344), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S92_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10240 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[576x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[576x1], 1]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile0, 1:[144x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[144x24], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[144x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[144x36], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 4]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile1, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+6640);
	KerArg0->W_In1 = (unsigned short int) (144);
	KerArg0->H_In1 = (unsigned short int) (24);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+576);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+10096);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+10192);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+10216);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6624);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6640), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+576), 5184, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10096), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10192), 24, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10216), 24, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6624), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*24);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+6624))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S93_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 2608 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+864);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+1728);
	KerArg0->Feat = (unsigned short int) (864);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+2592);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 864, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+864), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2592), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1728), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S96_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[96x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x1], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile1, 1:[24x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x144], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[1x144], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 4]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[36x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x144], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (24);
	KerArg0->H_In1 = (unsigned short int) (144);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+9600);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+3456);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+9216);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+9360);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+4032);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+9504);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+10464);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 576, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9216), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9360), 144, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10464), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+10464))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4032), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S99_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12544 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 144, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [D0, [0 x 576, 576]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 576, 576]]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1296 [D0, [0 x 1296, 1296]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1296, 1296]]
		Tile0: [0, 1296, 1296], Tile1: [0, 1296, 1296], Tile2; [0, 1296, 1296]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (144);
	KerArg0->OutFeatures = (unsigned short int) (144);
	KerArg0->TotalInFeatures = (unsigned short int) (144);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+6048);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+5184);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+5904);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+12528);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7344);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 5184, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5184), 576, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 144, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5904), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6048), 1296, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+12528), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+12528))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7344), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S102_Conv2d_24x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10240 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[576x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[576x1], 1]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile0, 1:[144x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[144x24], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[144x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[144x36], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[1x24], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 4]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile1, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 24 [Tile0, 1:[1x24], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x24], 1]
		Tile0: [0, 24, 24], Tile1: [0, 24, 24], Tile2; [0, 24, 24]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+6640);
	KerArg0->W_In1 = (unsigned short int) (144);
	KerArg0->H_In1 = (unsigned short int) (24);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+576);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+10096);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+10192);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+10216);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6624);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6640), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+576), 5184, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10096), 96, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10192), 24, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10216), 24, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6624), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*24);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+6624))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S103_MatAdd_24x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 2608 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[1x864], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x864], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+864);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+1728);
	KerArg0->Feat = (unsigned short int) (864);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+2592);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 864, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+864), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2592), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1728), 864, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S106_Conv2d_144x24x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 10480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 96 [Tile0, 1:[96x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[96x1], 1]
		Tile0: [0, 96, 96], Tile1: [0, 96, 96], Tile2; [0, 96, 96]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3456 [Tile1, 1:[24x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[24x144], 1]
		Tile0: [0, 3456, 3456], Tile1: [0, 3456, 3456], Tile2; [0, 3456, 3456]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 864 [Tile0, 1:[24x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[24x36], 1]
		Tile0: [0, 864, 864], Tile1: [0, 864, 864], Tile2; [0, 864, 864]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[1x144], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 4]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[36x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x144], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [Tile1, 1:[1x144], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x144], 1]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (24);
	KerArg0->H_In1 = (unsigned short int) (144);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+9600);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+3456);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+9216);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+9360);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+4032);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+9504);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+10464);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3456, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9600), 864, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 576, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9216), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9360), 144, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10464), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+10464))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4032), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S109_Conv2d_144x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 12544 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 144, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [D0, [0 x 576, 576]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 576, 576]]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 144 [D0, [0 x 144, 144]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 144, 144]]
		Tile0: [0, 144, 144], Tile1: [0, 144, 144], Tile2; [0, 144, 144]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1296 [D0, [0 x 1296, 1296]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1296, 1296]]
		Tile0: [0, 1296, 1296], Tile1: [0, 1296, 1296], Tile2; [0, 1296, 1296]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5184, 5184]][Tile0, 1:[6x6], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (144);
	KerArg0->OutFeatures = (unsigned short int) (144);
	KerArg0->TotalInFeatures = (unsigned short int) (144);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+6048);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+5184);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+5904);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+12528);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7344);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 5184, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5184), 576, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 144, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5904), 144, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6048), 1296, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+12528), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+12528))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7344), 5184, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S112_Conv2d_32x144x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 11728 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 576 [Tile1, 1:[576x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[576x1], 1]
		Tile0: [0, 576, 576], Tile1: [0, 576, 576], Tile2; [0, 576, 576]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4608 [Tile0, 1:[144x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[144x32], 1]
		Tile0: [0, 4608, 4608], Tile1: [0, 4608, 4608], Tile2; [0, 4608, 4608]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5184 [Tile1, 1:[144x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[144x36], 1]
		Tile0: [0, 5184, 5184], Tile1: [0, 5184, 5184], Tile2; [0, 5184, 5184]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[1x32], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 4]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile1, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+6928);
	KerArg0->W_In1 = (unsigned short int) (144);
	KerArg0->H_In1 = (unsigned short int) (32);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+576);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+11536);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+11664);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+11696);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+5760);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+6912);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6928), 4608, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+576), 5184, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11536), 128, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11664), 32, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11696), 32, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*32);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+6912))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5760), 1152, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S115_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 15504 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[128x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[128x1], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6144 [Tile1, 1:[32x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x192], 1]
		Tile0: [0, 6144, 6144], Tile1: [0, 6144, 6144], Tile2; [0, 6144, 6144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile1, 1:[1x192], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 4]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile1, 1:[36x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x192], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (32);
	KerArg0->H_In1 = (unsigned short int) (192);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+14336);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+6144);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+14016);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+15488);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6144, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14336), 1152, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6144), 768, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14016), 192, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15488), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+15488))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S118_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 16720 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 192, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1728 [D0, [0 x 1728, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1728, 1728]]
		Tile0: [0, 1728, 1728], Tile1: [0, 1728, 1728], Tile2; [0, 1728, 1728]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (192);
	KerArg0->OutFeatures = (unsigned short int) (192);
	KerArg0->TotalInFeatures = (unsigned short int) (192);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+7872);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+16704);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+9792);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6912, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 768, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7680), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7872), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 1728, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16704), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16704))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9792), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S121_Conv2d_32x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 15184 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile1, 1:[768x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[768x1], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6144 [Tile0, 1:[192x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[192x32], 1]
		Tile0: [0, 6144, 6144], Tile1: [0, 6144, 6144], Tile2; [0, 6144, 6144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile1, 1:[192x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[192x36], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[1x32], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 4]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile1, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+8848);
	KerArg0->W_In1 = (unsigned short int) (192);
	KerArg0->H_In1 = (unsigned short int) (32);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+768);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+14992);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+15120);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+15152);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+8832);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8848), 6144, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+768), 6912, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14992), 128, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15120), 32, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15152), 32, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8832), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*32);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+8832))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7680), 1152, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S122_MatAdd_32x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 3472 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+1152);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+2304);
	KerArg0->Feat = (unsigned short int) (1152);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+3456);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 1152, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1152), 1152, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2304), 1152, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S125_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 15504 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[128x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[128x1], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6144 [Tile1, 1:[32x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x192], 1]
		Tile0: [0, 6144, 6144], Tile1: [0, 6144, 6144], Tile2; [0, 6144, 6144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile1, 1:[1x192], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 4]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile1, 1:[36x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x192], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (32);
	KerArg0->H_In1 = (unsigned short int) (192);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+14336);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+6144);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+14016);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+15488);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6144, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14336), 1152, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6144), 768, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14016), 192, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15488), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+15488))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S128_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 16720 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 192, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1728 [D0, [0 x 1728, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1728, 1728]]
		Tile0: [0, 1728, 1728], Tile1: [0, 1728, 1728], Tile2; [0, 1728, 1728]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (192);
	KerArg0->OutFeatures = (unsigned short int) (192);
	KerArg0->TotalInFeatures = (unsigned short int) (192);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+7872);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+16704);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+9792);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6912, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 768, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7680), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7872), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 1728, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16704), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+16704))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9792), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S131_Conv2d_32x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 15184 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile1, 1:[768x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[768x1], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6144 [Tile0, 1:[192x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[192x32], 1]
		Tile0: [0, 6144, 6144], Tile1: [0, 6144, 6144], Tile2; [0, 6144, 6144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile1, 1:[192x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[192x36], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[1x32], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 4]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile1, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile1, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+8848);
	KerArg0->W_In1 = (unsigned short int) (192);
	KerArg0->H_In1 = (unsigned short int) (32);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+768);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+14992);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+15120);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+15152);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->ColFirst = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+8832);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8848), 6144, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+768), 6912, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14992), 128, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15120), 32, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15152), 32, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8832), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*32);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+8832))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7680), 1152, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S132_MatAdd_32x6x6(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 3472 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[1x1152], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1152], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+1152);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+2304);
	KerArg0->Feat = (unsigned short int) (1152);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+3456);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 1152, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1152), 1152, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3456), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+2304), 1152, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S135_Conv2d_192x32x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 15504 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [Tile0, 1:[128x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[128x1], 1]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6144 [Tile1, 1:[32x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[32x192], 1]
		Tile0: [0, 6144, 6144], Tile1: [0, 6144, 6144], Tile2; [0, 6144, 6144]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1152 [Tile0, 1:[32x36], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[32x36], 1]
		Tile0: [0, 1152, 1152], Tile1: [0, 1152, 1152], Tile2; [0, 1152, 1152]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile1, 1:[1x192], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 4]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [Tile1, 1:[36x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[36x192], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [Tile1, 1:[1x192], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x192], 1]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (32);
	KerArg0->H_In1 = (unsigned short int) (192);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+14336);
	KerArg0->W_In2 = (unsigned short int) (36);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+6144);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+13824);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+14016);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->W_Out = (unsigned short int) (36);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+14208);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+15488);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6144, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14336), 1152, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6144), 768, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+13824), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14016), 192, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+15488), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*36);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+15488))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 6912, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S138_Conv2d_192x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 11536 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 192, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 6912 [D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 6912, 6912]][Tile0, 1:[6x6], 1]
		Tile0: [0, 6912, 6912], Tile1: [0, 6912, 6912], Tile2; [0, 6912, 6912]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 192 [D0, [0 x 192, 192]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 192, 192]]
		Tile0: [0, 192, 192], Tile1: [0, 192, 192], Tile2; [0, 192, 192]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1728 [D0, [0 x 1728, 1728]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1728, 1728]]
		Tile0: [0, 1728, 1728], Tile1: [0, 1728, 1728], Tile2; [0, 1728, 1728]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1728 [D0, [0 x 1728, 1728]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1728, 1728]][Tile0, 1:[3x3], 1]
		Tile0: [0, 1728, 1728], Tile1: [0, 1728, 1728], Tile2; [0, 1728, 1728]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (6);
	KerArg0->UsedW = (unsigned short int) (6);
	KerArg0->H = (unsigned short int) (6);
	KerArg0->InFeatures = (unsigned short int) (192);
	KerArg0->OutFeatures = (unsigned short int) (192);
	KerArg0->TotalInFeatures = (unsigned short int) (192);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+6912);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+7680);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+7872);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+11520);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+9792);
	KerArg0->Pad = (v4s) ((v4s){0,1,0,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 6912, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+6912), 768, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7680), 192, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+7872), 192, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 1728, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11520), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (6);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+11520))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride2B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+9792), 1728, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S141_Conv2d_56x192x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [Tile0, 1:[768x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[768x1], 1]
		Tile0: [0, 768, 768], Tile1: [0, 768, 768], Tile2; [0, 768, 768]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10752 [Tile1, 1:[192x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[192x56], 1]
		Tile0: [0, 10752, 10752], Tile1: [0, 10752, 10752], Tile2; [0, 10752, 10752]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1728 [Tile0, 1:[192x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[192x9], 1]
		Tile0: [0, 1728, 1728], Tile1: [0, 1728, 1728], Tile2; [0, 1728, 1728]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile1, 1:[1x56], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 4]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile1, 1:[9x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x56], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (192);
	KerArg0->H_In1 = (unsigned short int) (56);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+12360);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+10752);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+11480);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+11536);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+10976);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+11592);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+14088);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 10752, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+12360), 1728, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10752), 224, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11480), 56, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11536), 56, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+14088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+14088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+10976), 504, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S144_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24600 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile0, 1:[224x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[224x1], 1]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 18816 [Tile1, 1:[56x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[56x336], 1]
		Tile0: [0, 18816, 18816], Tile1: [0, 18816, 18816], Tile2; [0, 18816, 18816]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[56x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[56x9], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile1, 1:[1x336], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 4]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile1, 1:[9x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x336], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (56);
	KerArg0->H_In1 = (unsigned short int) (336);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+24080);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18816);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+23184);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+23520);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+20160);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+23856);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+24584);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 18816, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24080), 504, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18816), 1344, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23184), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23520), 336, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24584), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+24584))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+20160), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S147_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 11104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 336, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [D0, [0 x 1344, 1344]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1344, 1344]]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (3);
	KerArg0->UsedW = (unsigned short int) (3);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->InFeatures = (unsigned short int) (336);
	KerArg0->OutFeatures = (unsigned short int) (336);
	KerArg0->TotalInFeatures = (unsigned short int) (336);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+5040);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+3024);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+4368);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+4704);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+11088);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3024, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3024), 1344, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4368), 336, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4704), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5040), 3024, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (3);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+11088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S150_Conv2d_56x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24040 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile0, 1:[1344x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1344x1], 1]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 18816 [Tile1, 1:[336x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[336x56], 1]
		Tile0: [0, 18816, 18816], Tile1: [0, 18816, 18816], Tile2; [0, 18816, 18816]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile0, 1:[336x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[336x9], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile1, 1:[1x56], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 4]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile1, 1:[9x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x56], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (336);
	KerArg0->H_In1 = (unsigned short int) (56);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+21000);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18816);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+19544);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+19600);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+19040);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+19656);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+24024);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 18816, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+21000), 3024, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18816), 224, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19544), 56, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19600), 56, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24024), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+24024))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19040), 504, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S151_MatAdd_56x3x3(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 1528 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+504);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+1008);
	KerArg0->Feat = (unsigned short int) (504);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+1512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 504, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+504), 504, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1512), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1008), 504, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S154_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24600 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile0, 1:[224x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[224x1], 1]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 18816 [Tile1, 1:[56x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[56x336], 1]
		Tile0: [0, 18816, 18816], Tile1: [0, 18816, 18816], Tile2; [0, 18816, 18816]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[56x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[56x9], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile1, 1:[1x336], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 4]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile1, 1:[9x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x336], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (56);
	KerArg0->H_In1 = (unsigned short int) (336);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+24080);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18816);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+23184);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+23520);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+20160);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+23856);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+24584);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 18816, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24080), 504, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18816), 1344, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23184), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23520), 336, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24584), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+24584))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+20160), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S157_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 11104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 336, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [D0, [0 x 1344, 1344]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1344, 1344]]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (3);
	KerArg0->UsedW = (unsigned short int) (3);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->InFeatures = (unsigned short int) (336);
	KerArg0->OutFeatures = (unsigned short int) (336);
	KerArg0->TotalInFeatures = (unsigned short int) (336);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+5040);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+3024);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+4368);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+4704);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+11088);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3024, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3024), 1344, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4368), 336, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4704), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5040), 3024, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (3);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+11088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S160_Conv2d_56x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24040 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile0, 1:[1344x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1344x1], 1]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 18816 [Tile1, 1:[336x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[336x56], 1]
		Tile0: [0, 18816, 18816], Tile1: [0, 18816, 18816], Tile2; [0, 18816, 18816]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile0, 1:[336x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[336x9], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile1, 1:[1x56], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 4]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile1, 1:[9x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x56], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 56 [Tile1, 1:[1x56], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x56], 1]
		Tile0: [0, 56, 56], Tile1: [0, 56, 56], Tile2; [0, 56, 56]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (336);
	KerArg0->H_In1 = (unsigned short int) (56);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+21000);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18816);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+19544);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+19600);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+19040);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+19656);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+24024);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 18816, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+21000), 3024, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18816), 224, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19544), 56, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19600), 56, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24024), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+24024))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19040), 504, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S161_MatAdd_56x3x3(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 1528 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[1x504], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x504], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (classification_L1_Memory+504);
	KerArg0->Out = (signed char *__restrict__) (classification_L1_Memory+1008);
	KerArg0->Feat = (unsigned short int) (504);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+1512);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 504, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+504), 504, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1512), 13, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerMatAdd_SQ8, (void *) KerArg0);
		__CALL(KerMatAdd_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1008), 504, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S164_Conv2d_336x56x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24600 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 1][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 224 [Tile0, 1:[224x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[224x1], 1]
		Tile0: [0, 224, 224], Tile1: [0, 224, 224], Tile2; [0, 224, 224]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 18816 [Tile1, 1:[56x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[56x336], 1]
		Tile0: [0, 18816, 18816], Tile1: [0, 18816, 18816], Tile2; [0, 18816, 18816]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 504 [Tile0, 1:[56x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[56x9], 1]
		Tile0: [0, 504, 504], Tile1: [0, 504, 504], Tile2; [0, 504, 504]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile1, 1:[1x336], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 4]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile1, 1:[9x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[9x336], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [Tile1, 1:[1x336], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 1:[1x336], 1]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W_In1 = (unsigned short int) (56);
	KerArg0->H_In1 = (unsigned short int) (336);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+24080);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+18816);
	KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+23184);
	KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+23520);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+20160);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+23856);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+24584);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 18816, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24080), 504, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+18816), 1344, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23184), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+23520), 336, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+24584), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile1 */
		int T1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+24584))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+20160), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S167_Conv2d_336x1x3x3_Relu6(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 11104 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 336, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [D0, [0 x 1344, 1344]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1344, 1344]]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 336 [D0, [0 x 336, 336]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 336, 336]]
		Tile0: [0, 336, 336], Tile1: [0, 336, 336], Tile2; [0, 336, 336]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 3024, 3024]][Tile0, 1:[3x3], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (3);
	KerArg0->UsedW = (unsigned short int) (3);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->InFeatures = (unsigned short int) (336);
	KerArg0->OutFeatures = (unsigned short int) (336);
	KerArg0->TotalInFeatures = (unsigned short int) (336);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+5040);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+3024);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+4368);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+4704);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+11088);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+8064);
	KerArg0->Pad = (v4s) ((v4s){1,1,1,1});
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 3024, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+3024), 1344, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4368), 336, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4704), 336, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+5040), 3024, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11088), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (3);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+11088))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8064), 3024, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S170_Conv2d_112x336x1x1(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 38176 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Total=0, T1Ind_Last, T1Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 3][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1344 [Tile0, 1:[1344x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1344x1], 1]
		Tile0: [0, 1344, 1344], Tile1: [0, 1344, 1344], Tile2; [0, 1344, 1344]
	Ker Arg: In1, Tiled Space: Tile1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 37632 [Tile1, 3:[336x48, 1:336x48, 336x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 3:[336x48, 1:336x48, 336x16], 1]
		Tile0: [0, 16128, 16128], Tile1: [16128, 16128, 16128], Tile2; [32256, 5376, 5376]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 3024 [Tile0, 1:[336x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[336x9], 1]
		Tile0: [0, 3024, 3024], Tile1: [0, 3024, 3024], Tile2; [0, 3024, 3024]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 448 [Tile1, 3:[1x48, 1:1x48, 1x16], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 3:[1x48, 1:1x48, 1x16], 4]
		Tile0: [0, 448, 448], Tile1: [0, 448, 448], Tile2; [0, 448, 448]
	Ker Arg: Out, Tiled Space: Tile1
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 1008 [Tile1, 3:[9x48, 1:9x48, 9x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 3:[9x48, 1:9x48, 9x16], 1]
		Tile0: [0, 432, 432], Tile1: [432, 432, 432], Tile2; [864, 144, 144]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 112 [Tile1, 3:[1x48, 1:1x48, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 3:[1x48, 1:1x48, 1x16], 1]
		Tile0: [0, 112, 112], Tile1: [0, 112, 112], Tile2; [0, 112, 112]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 3 logical tiles, 1 physical tiles
			Total Size: 112 [Tile1, 3:[1x48, 1:1x48, 1x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 3:[1x48, 1:1x48, 1x16], 1]
		Tile0: [0, 112, 112], Tile1: [0, 112, 112], Tile2; [0, 112, 112]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W_In1 = (unsigned short int) (336);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+35136);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+33792);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+38160);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 16128, 0, DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+35136), 3024, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32256), 448, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=432;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+33568), 112, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+33680), 112, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+38160), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T1Ind=0; T1Ind<3; T1Ind++, T1Ind_Total++) { /* Iteration on Tile1 */
		int T1Ind_Last = (T1Ind==2), T1Ind_NextLast = ((T1Ind+1)==2);
		/*================================= Prepare Tiles ===================================*/
		_SN_In1 = 0;
		if (!(T1Ind_Last)) {
			_N_In1 = _N_In1 + (16128); _SN_In1 = ((T1Ind_NextLast)?5376:16128); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
		if (_SN_In1) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+16128*((T1Ind_Total+1)%2)),
					_SN_In1, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0+16128*((T1Ind_Total)%2));
			KerArg0->H_In1 = (unsigned short int) (T1Ind_Last?16:48);
			KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+32256+(192*(T1Ind)));
			KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+33568+(48*(T1Ind)));
			KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+33680+(48*(T1Ind)));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+32704+432*((T1Ind_Total)%2));
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+38160))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32704+432*((T1Ind_Total)%2)),
				_SC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T1Ind_Last)) {
			_C_Out = _C_Out + (432); _SC_Out = ((T1Ind_NextLast)?144:432); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S173_Conv2d_1280x112x1x1_Relu6(
		signed char * __restrict__ In2,
		signed char * __restrict__ In1,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 44000 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerMatMul_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T1Ind, T1Ind_Total=0, T1Ind_Last, T1Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_In1;
	unsigned int _SN_In1;
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile1 Dim: 9][Tile0 Dim: 1]
	Ker Arg: KerBuff, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 448 [Tile0, 1:[448x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[448x1], 1]
		Tile0: [0, 448, 448], Tile1: [0, 448, 448], Tile2; [0, 448, 448]
	Ker Arg: In1, Tiled Space: Tile1
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 9 logical tiles, 9 physical tiles
			Total Size: 143360 [Tile1, 9:[112x144, 7:112x144, 112x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 9:[112x144, 7:112x144, 112x128], 1]
		Tile0: [0, 16128, 16128], Tile1: [16128, 16128, 16128], Tile2; [32256, 16128, 16128]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1008 [Tile0, 1:[112x9], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[112x9], 1]
		Tile0: [0, 1008, 1008], Tile1: [0, 1008, 1008], Tile2; [0, 1008, 1008]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 9 logical tiles, 1 physical tiles
			Total Size: 5120 [Tile1, 9:[1x144, 7:1x144, 1x128], 4]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 9:[1x144, 7:1x144, 1x128], 4]
		Tile0: [0, 5120, 5120], Tile1: [0, 5120, 5120], Tile2; [0, 5120, 5120]
	Ker Arg: Out, Tiled Space: Tile1
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 9 logical tiles, 9 physical tiles
			Total Size: 11520 [Tile1, 9:[9x144, 7:9x144, 9x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 9:[9x144, 7:9x144, 9x128], 1]
		Tile0: [0, 1296, 1296], Tile1: [1296, 1296, 1296], Tile2; [2592, 1296, 1296]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 9 logical tiles, 1 physical tiles
			Total Size: 1280 [Tile1, 9:[1x144, 7:1x144, 1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 9:[1x144, 7:1x144, 1x128], 1]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 9 logical tiles, 1 physical tiles
			Total Size: 1280 [Tile1, 9:[1x144, 7:1x144, 1x128], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile1, 9:[1x144, 7:1x144, 1x128], 1]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->W_In1 = (unsigned short int) (112);
	KerArg0->In2 = (signed char * __restrict__) (classification_L1_Memory+42976);
	KerArg0->W_In2 = (unsigned short int) (9);
	KerArg0->W_Out = (unsigned short int) (9);
	KerArg0->BufferColIn2 = (signed char * __restrict__) (classification_L1_Memory+42528);
	KerArg0->ColFirst = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+43984);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+0), 16128, 0, DmaR_Evt1);
	_N_In1=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42976), 1008, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32256), 5120, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	_C_Out=0; _SC_Out=1296;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+39968), 1280, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+41248), 1280, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+43984), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (T1Ind=0; T1Ind<9; T1Ind++, T1Ind_Total++) { /* Iteration on Tile1 */
		int T1Ind_Last = (T1Ind==8), T1Ind_NextLast = ((T1Ind+1)==8);
		/*================================= Prepare Tiles ===================================*/
		_SN_In1 = 0;
		if (!(T1Ind_Last)) {
			_N_In1 = _N_In1 + (16128); _SN_In1 = ((T1Ind_NextLast)?14336:16128); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In1 */
		if (_SN_In1) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+_N_In1), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0+16128*((T1Ind_Total+1)%2)),
					_SN_In1, 0, DmaR_Evt1);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->In1 = (signed char * __restrict__) (classification_L1_Memory+0+16128*((T1Ind_Total)%2));
			KerArg0->H_In1 = (unsigned short int) (T1Ind_Last?128:144);
			KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+32256+(576*(T1Ind)));
			KerArg0->Scale = (unsigned char * __restrict__) (classification_L1_Memory+39968+(144*(T1Ind)));
			KerArg0->ScaleN = (unsigned char * __restrict__) (classification_L1_Memory+41248+(144*(T1Ind)));
			KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+37376+1296*((T1Ind_Total)%2));
			KerArg0->OutFirstCol = (unsigned short int) ((0)*9);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+43984))[8]);
			AT_FORK(gap_ncore(), (void *) KerParMatMulB32_ReLUM_SQ8, (void *) KerArg0);
			__CALL(KerParMatMulB32_ReLUM_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Write Tiles =====================================*/
		if (_SP_Out) AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
		AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+37376+1296*((T1Ind_Total)%2)),
				_SC_Out, 1, DmaW_Evt1);
		/*============================= End Write Tiles =====================================*/
		/*================================= Update Arg Pipeline =============================*/
		_SP_Out = _SC_Out;
		/*============================= End Update Arg Pipeline =============================*/
		/*================================= Prepare Tiles ===================================*/
		_SC_Out = 0;
		if (!(T1Ind_Last)) {
			_C_Out = _C_Out + (1296); _SC_Out = ((T1Ind_NextLast)?1152:1296); 
		}
		/*============================= End Prepare Tiles ===================================*/
	} /* End iteration on Tile1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S176_Conv2d_1280x1x3x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 32016 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerConvRedAct_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 1280, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 11520 [D0, [0 x 11520, 11520]][Tile0, 1:[3x3], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 11520, 11520]][Tile0, 1:[3x3], 1]
		Tile0: [0, 11520, 11520], Tile1: [0, 11520, 11520], Tile2; [0, 11520, 11520]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5120 [D0, [0 x 5120, 5120]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 5120, 5120]]
		Tile0: [0, 5120, 5120], Tile1: [0, 5120, 5120], Tile2; [0, 5120, 5120]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1280 [D0, [0 x 1280, 1280]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1280, 1280]]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1280 [D0, [0 x 1280, 1280]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1280, 1280]]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 11520 [D0, [0 x 11520, 11520]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 11520, 11520]]
		Tile0: [0, 11520, 11520], Tile1: [0, 11520, 11520], Tile2; [0, 11520, 11520]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1280 [D0, [0 x 1280, 1280]][Tile0, 1:[1x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 1280, 1280]][Tile0, 1:[1x1], 1]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (3);
	KerArg0->UsedW = (unsigned short int) (3);
	KerArg0->H = (unsigned short int) (3);
	KerArg0->InFeatures = (unsigned short int) (1280);
	KerArg0->OutFeatures = (unsigned short int) (1280);
	KerArg0->TotalInFeatures = (unsigned short int) (1280);
	KerArg0->Filter = (signed char * __restrict__) (classification_L1_Memory+19200);
	KerArg0->Bias = (signed char * __restrict__) (classification_L1_Memory+11520);
	KerArg0->Scale = (signed char * __restrict__) (classification_L1_Memory+16640);
	KerArg0->ScaleN = (signed char * __restrict__) (classification_L1_Memory+17920);
	KerArg0->Infos = (signed char * __restrict__) (classification_L1_Memory+32000);
	KerArg0->Out = (signed char * __restrict__) (classification_L1_Memory+30720);
	KerArg0->Pad = (v4s) 0;
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 11520, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+11520), 5120, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+16640), 1280, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+17920), 1280, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Filter+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+12800+0), 11520, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read Filter */
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+12800+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+19200), 11520, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32000), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->UsedH = (unsigned short int) (3);
			KerArg0->NormBias = (unsigned char) (((char *)(classification_L1_Memory+32000))[8]);
			AT_FORK(gap_ncore(), (void *) KerParConvDWRedAct3x3Stride1B32_SQ8, (void *) KerArg0);
			__CALL(KerParConvDWRedAct3x3Stride1B32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+30720), 1280, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S179_Conv2d_32x1280x1x1_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 42480 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	KerLinear_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 32, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1280 [Tile0, 1:[1x1], 1280]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 1280]
		Tile0: [0, 1280, 1280], Tile1: [0, 1280, 1280], Tile2; [0, 1280, 1280]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 40960 [D0, [0 x 40960, 40960]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 40960, 40960]]
		Tile0: [0, 40960, 40960], Tile1: [0, 40960, 40960], Tile2; [0, 40960, 40960]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D0, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D0, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D0, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D0, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->Weights = (signed char * __restrict__) (classification_L1_Memory+1280);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+42240);
	KerArg0->Out = (void * __restrict__) (classification_L1_Memory+42368);
	KerArg0->InDim = (unsigned short int) (1280);
	KerArg0->TotalInDim = (unsigned short int) (1280);
	KerArg0->OutDim = (unsigned short int) (32);
	KerArg0->Scale = (unsigned char *__restrict__) (classification_L1_Memory+42400);
	KerArg0->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+42432);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+42464);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 1280, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+1280), 40960, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42240), 128, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42400), 32, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42432), 32, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42464), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerFullFeatB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParLinearLayerFullFeatB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+42368), 32, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S180_Op_MEAN_0_72(
		signed char * __restrict__ In,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 80 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerGlobalPool_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x32], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x32], 1]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (void * __restrict__) (classification_L1_Memory+0);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Out = (int * __restrict__) (classification_L1_Memory+32);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (void * __restrict__) (classification_L1_Memory+64);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 32, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Infos+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+32+0), 13, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read Infos */
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+32+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+64), 13, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) KerParGlobalAvgPoolFullFeat_SQ8, (void *) KerArg0);
		__CALL(KerParGlobalAvgPoolFullFeat_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32), 32, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S183_Linear_2x32(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 132 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_L2_EVENT _DmaR_Evt3, *DmaR_Evt3 = &_DmaR_Evt3;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt4, *DmaR_Evt4 = &_DmaR_Evt4;
	AT_L2_EVENT _DmaR_Evt5, *DmaR_Evt5 = &_DmaR_Evt5;
	AT_L2_EVENT _DmaR_Evt6, *DmaR_Evt6 = &_DmaR_Evt6;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	AT_HYPERRAM_CL_EVENT _UchanHR2, *UchanHR2 = &_UchanHR2;
	KerLinear_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 2, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [Tile0, 1:[1x1], 32]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 32]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D0, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 8 [D0, [0 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 8, 8]]
		Tile0: [0, 8, 8], Tile1: [0, 8, 8], Tile2; [0, 8, 8]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2 [D0, [0 x 2, 2]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2, 2]]
		Tile0: [0, 2, 2], Tile1: [0, 2, 2], Tile2; [0, 2, 2]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2 [D0, [0 x 2, 2]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2, 2]]
		Tile0: [0, 2, 2], Tile1: [0, 2, 2], Tile2; [0, 2, 2]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2 [D0, [0 x 2, 2]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 2, 2]]
		Tile0: [0, 2, 2], Tile1: [0, 2, 2], Tile2; [0, 2, 2]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[1x1], 13]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 13]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (classification_L1_Memory+0);
	KerArg0->Weights = (signed char * __restrict__) (classification_L1_Memory+32);
	KerArg0->Bias = (void * __restrict__) (classification_L1_Memory+96);
	KerArg0->Out = (void * __restrict__) (classification_L1_Memory+104);
	KerArg0->InDim = (unsigned short int) (32);
	KerArg0->TotalInDim = (unsigned short int) (32);
	KerArg0->OutDim = (unsigned short int) (2);
	KerArg0->Scale = (unsigned char *__restrict__) (classification_L1_Memory+108);
	KerArg0->ScaleN = (unsigned char *__restrict__) (classification_L1_Memory+112);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+116);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 32, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Filter+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+36+0), 64, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read Filter */
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+36+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+32), 64, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+96), 8, 0, DmaR_Evt3);
	AT_L2_WAIT(0, DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+108), 2, 0, DmaR_Evt4);
	AT_L2_WAIT(0, DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+112), 2, 0, DmaR_Evt5);
	AT_L2_WAIT(0, DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Infos+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+100+0), 13, 0, UchanHR2);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2); /* Wait previous uDMA read Infos */
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+100+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+116), 13, 0, DmaR_Evt6);
	AT_L2_WAIT(0, DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerFullFeatB32_SQ8, (void *) KerArg0);
			__CALL(KerParLinearLayerFullFeatB32_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+104), 2, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S184_SoftMax(
		signed char * __restrict__ In,
		short int * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 24 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT _DmaR_Evt1, *DmaR_Evt1 = &_DmaR_Evt1;
	AT_L2_EVENT _DmaW_Evt1, *DmaW_Evt1 = &_DmaW_Evt1;
	AT_L2_EVENT _DmaR_Evt2, *DmaR_Evt2 = &_DmaR_Evt2;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	KerSoftMax_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 2 [Tile0, 1:[2x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[2x1], 1]
		Tile0: [0, 2, 2], Tile1: [0, 2, 2], Tile2; [0, 2, 2]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4 [Tile0, 1:[2x1], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[2x1], 2]
		Tile0: [0, 4, 4], Tile1: [0, 4, 4], Tile2; [0, 4, 4]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 13 [Tile0, 1:[13x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[13x1], 1]
		Tile0: [0, 13, 13], Tile1: [0, 13, 13], Tile2; [0, 13, 13]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (classification_L1_Memory+0);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->N = (unsigned short int) (2);
	KerArg0->Out = (short int *__restrict__) (classification_L1_Memory+4);
	KerArg0->Infos = (signed char *__restrict__) (classification_L1_Memory+8);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+0), 2, 0, DmaR_Evt1);
	AT_L2_WAIT(0, DmaR_Evt1); /* Wait previous DMA read In */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) Infos+0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+0), 13, 0, UchanHR1);
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1); /* Wait previous uDMA read Infos */
	AT_L2_COPY(0, ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn+0+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+8), 13, 0, DmaR_Evt2);
	AT_L2_WAIT(0, DmaR_Evt2); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->Norm = (unsigned short int) (((char *)(classification_L1_Memory+8))[0]);
		AT_FORK(gap_ncore(), (void *) KerParSoftMax_SQ8, (void *) KerArg0);
		__CALL(KerParSoftMax_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) classification_L1_Memory+4), 4, 1, DmaW_Evt1);
	AT_L2_WAIT(0, DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int classificationCNN_Construct()

{
	AT_HYPERRAM_CONF_T HyperRamConf;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;

	int Error;
	AT_HYPERRAM_CONF_INIT(&HyperRamConf, AT_MEM_L3_HRAM, 0);
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERRAM_OPEN(&HyperRam, &HyperRamConf, &Error);
	if (Error) return 1;
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "classification_L3_Flash_Const.dat", &Error);
	if (Error) return 1;

	classification_L3_Memory = (AT_HYPERRAM_POINTER) AT_HYPERRAM_ALLOC(&HyperRam, 461168);
	if (classification_L3_Memory == 0) return 2;
	classification_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 76208);
	if (classification_L2_Memory == 0) return 3;
	classification_L2_Memory_Dyn = (AT_L2_POINTER) AT_L2_ALLOC(0, 193792);
	if (classification_L2_Memory_Dyn == 0) return 3;
	classification_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 46696);
	if (classification_L1_Memory == 0) return 4;
	AT_HYPERFLASH_FS_FC_EVENT _UchanHF1, *UchanHF1 = &_UchanHF1;
	AT_HYPERRAM_FC_EVENT _UchanHR2, *UchanHR2 = &_UchanHR2;
	/* Moving Separable_conv2ddepthwise_kern, size 1 from HyperFlash at 485828 to (size 1) HyperRam at 461132..461132 */
	{
		int Size = 1, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485828 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461132 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialseparable_conv2dsepa, size 4 from HyperFlash at 485832 to (size 4) HyperRam at 461136..461139 */
	{
		int Size = 4, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485832 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461136 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S3_Mul_scale, size 1 from HyperFlash at 485836 to (size 1) HyperRam at 461140..461140 */
	{
		int Size = 1, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485836 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461140 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S3_Mul_shift, size 1 from HyperFlash at 485840 to (size 1) HyperRam at 461144..461144 */
	{
		int Size = 1, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485840 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461144 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S3_Infos, size 13 from HyperFlash at 484496 to (size 13) HyperRam at 459800..459812 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484496 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459800 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialseparable_conv2dsepa_411470e1, size 3 from HyperFlash at 485844 to (size 3) HyperRam at 461148..461150 */
	{
		int Size = 3, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485844 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461148 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialseparable_conv2dbias, size 12 from HyperFlash at 485760 to (size 12) HyperRam at 461064..461075 */
	{
		int Size = 12, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485760 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461064 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S6_Mul_scale, size 3 from HyperFlash at 485848 to (size 3) HyperRam at 461152..461154 */
	{
		int Size = 3, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485848 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461152 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S6_Mul_shift, size 3 from HyperFlash at 485852 to (size 3) HyperRam at 461156..461158 */
	{
		int Size = 3, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485852 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461156 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S6_Infos, size 13 from HyperFlash at 484512 to (size 13) HyperRam at 459816..459828 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484512 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459816 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96co, size 432 from HyperFlash at 463696 to (size 432) HyperRam at 439168..439599 */
	{
		int Size = 432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 463696 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 439168 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S10_Mul_scale, size 16 from HyperFlash at 484528 to (size 16) HyperRam at 459832..459847 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484528 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459832 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S10_Mul_shift, size 16 from HyperFlash at 484544 to (size 16) HyperRam at 459848..459863 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484544 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459848 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S10_Infos, size 13 from HyperFlash at 484560 to (size 13) HyperRam at 459864..459876 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484560 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459864 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96ex_2e8d685a, size 64 from HyperFlash at 482720 to (size 64) HyperRam at 458048..458111 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482720 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458048 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S13_Mul_scale, size 16 from HyperFlash at 484576 to (size 16) HyperRam at 459880..459895 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484576 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459880 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S13_Mul_shift, size 16 from HyperFlash at 484592 to (size 16) HyperRam at 459896..459911 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484592 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459896 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S13_Infos, size 13 from HyperFlash at 484608 to (size 13) HyperRam at 459912..459924 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484608 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459912 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96ex_4dd58ac9, size 128 from HyperFlash at 480256 to (size 128) HyperRam at 455584..455711 */
	{
		int Size = 128, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480256 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455584 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96ex_ec77a7bf, size 32 from HyperFlash at 483952 to (size 32) HyperRam at 459280..459311 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483952 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459280 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S16_Mul_scale, size 8 from HyperFlash at 485772 to (size 8) HyperRam at 461076..461083 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485772 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461076 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S16_Mul_shift, size 8 from HyperFlash at 485780 to (size 8) HyperRam at 461084..461091 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485780 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461084 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S16_Infos, size 13 from HyperFlash at 484624 to (size 13) HyperRam at 459928..459940 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484624 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459928 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_ecf73e9c, size 384 from HyperFlash at 465424 to (size 384) HyperRam at 440896..441279 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 465424 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440896 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_d7b8fcc3, size 192 from HyperFlash at 474352 to (size 192) HyperRam at 449824..450015 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 474352 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449824 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S19_Mul_scale, size 48 from HyperFlash at 483376 to (size 48) HyperRam at 458704..458751 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483376 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458704 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S19_Mul_shift, size 48 from HyperFlash at 483424 to (size 48) HyperRam at 458752..458799 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483424 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458752 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S19_Infos, size 13 from HyperFlash at 484640 to (size 13) HyperRam at 459944..459956 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484640 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459944 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_5aec7246, size 432 from HyperFlash at 464128 to (size 432) HyperRam at 439600..440031 */
	{
		int Size = 432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 464128 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 439600 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_96627e8b, size 192 from HyperFlash at 474544 to (size 192) HyperRam at 450016..450207 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 474544 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450016 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S22_Mul_scale, size 48 from HyperFlash at 483472 to (size 48) HyperRam at 458800..458847 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483472 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458800 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S22_Mul_shift, size 48 from HyperFlash at 483520 to (size 48) HyperRam at 458848..458895 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483520 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458848 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S22_Infos, size 13 from HyperFlash at 484656 to (size 13) HyperRam at 459960..459972 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484656 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459960 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_cb056cce, size 384 from HyperFlash at 465808 to (size 384) HyperRam at 441280..441663 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 465808 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 441280 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_adb8c218, size 32 from HyperFlash at 483984 to (size 32) HyperRam at 459312..459343 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483984 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459312 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S25_Mul_scale, size 8 from HyperFlash at 485788 to (size 8) HyperRam at 461092..461099 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485788 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461092 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S25_Mul_shift, size 8 from HyperFlash at 485796 to (size 8) HyperRam at 461100..461107 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485796 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461100 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S25_Infos, size 13 from HyperFlash at 484672 to (size 13) HyperRam at 459976..459988 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484672 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459976 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_fc011502, size 384 from HyperFlash at 466192 to (size 384) HyperRam at 441664..442047 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 466192 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 441664 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_2be52376, size 192 from HyperFlash at 474736 to (size 192) HyperRam at 450208..450399 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 474736 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450208 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S28_Mul_scale, size 48 from HyperFlash at 483568 to (size 48) HyperRam at 458896..458943 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483568 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458896 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S28_Mul_shift, size 48 from HyperFlash at 483616 to (size 48) HyperRam at 458944..458991 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483616 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458944 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S28_Infos, size 13 from HyperFlash at 484688 to (size 13) HyperRam at 459992..460004 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484688 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459992 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_4d2ce9de, size 432 from HyperFlash at 464560 to (size 432) HyperRam at 440032..440463 */
	{
		int Size = 432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 464560 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440032 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_2448ea8a, size 192 from HyperFlash at 474928 to (size 192) HyperRam at 450400..450591 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 474928 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450400 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S31_Mul_scale, size 48 from HyperFlash at 483664 to (size 48) HyperRam at 458992..459039 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483664 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458992 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S31_Mul_shift, size 48 from HyperFlash at 483712 to (size 48) HyperRam at 459040..459087 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483712 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459040 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S31_Infos, size 13 from HyperFlash at 484704 to (size 13) HyperRam at 460008..460020 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484704 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460008 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9757a219, size 384 from HyperFlash at 466576 to (size 384) HyperRam at 442048..442431 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 466576 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442048 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_fc9e5e40, size 32 from HyperFlash at 484016 to (size 32) HyperRam at 459344..459375 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484016 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459344 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S34_Mul_scale, size 8 from HyperFlash at 485804 to (size 8) HyperRam at 461108..461115 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485804 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461108 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S34_Mul_shift, size 8 from HyperFlash at 485812 to (size 8) HyperRam at 461116..461123 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485812 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461116 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S34_Infos, size 13 from HyperFlash at 484720 to (size 13) HyperRam at 460024..460036 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484720 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460024 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S35_Infos, size 13 from HyperFlash at 484736 to (size 13) HyperRam at 460040..460052 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484736 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460040 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_4b74c721, size 384 from HyperFlash at 466960 to (size 384) HyperRam at 442432..442815 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 466960 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442432 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9be0ce95, size 192 from HyperFlash at 475120 to (size 192) HyperRam at 450592..450783 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 475120 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450592 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S38_Mul_scale, size 48 from HyperFlash at 483760 to (size 48) HyperRam at 459088..459135 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483760 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459088 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S38_Mul_shift, size 48 from HyperFlash at 483808 to (size 48) HyperRam at 459136..459183 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483808 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459136 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S38_Infos, size 13 from HyperFlash at 484752 to (size 13) HyperRam at 460056..460068 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484752 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460056 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_d944b225, size 432 from HyperFlash at 464992 to (size 432) HyperRam at 440464..440895 */
	{
		int Size = 432, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 464992 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440464 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_22a2b491, size 192 from HyperFlash at 475312 to (size 192) HyperRam at 450784..450975 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 475312 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450784 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S41_Mul_scale, size 48 from HyperFlash at 483856 to (size 48) HyperRam at 459184..459231 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483856 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459184 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S41_Mul_shift, size 48 from HyperFlash at 483904 to (size 48) HyperRam at 459232..459279 */
	{
		int Size = 48, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483904 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459232 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S41_Infos, size 13 from HyperFlash at 484768 to (size 13) HyperRam at 460072..460084 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484768 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460072 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_b7f5ec57, size 768 from HyperFlash at 453264 to (size 768) HyperRam at 428736..429503 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 453264 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 428736 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_1f34f003, size 64 from HyperFlash at 482784 to (size 64) HyperRam at 458112..458175 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482784 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458112 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S44_Mul_scale, size 16 from HyperFlash at 484784 to (size 16) HyperRam at 460088..460103 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484784 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460088 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S44_Mul_shift, size 16 from HyperFlash at 484800 to (size 16) HyperRam at 460104..460119 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484800 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460104 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S44_Infos, size 13 from HyperFlash at 484816 to (size 13) HyperRam at 460120..460132 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484816 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460120 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_6437f573, size 384 from HyperFlash at 467344 to (size 384) HyperRam at 442816..443199 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 467344 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442816 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S47_Mul_scale, size 96 from HyperFlash at 481120 to (size 96) HyperRam at 456448..456543 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481120 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456448 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S47_Mul_shift, size 96 from HyperFlash at 481216 to (size 96) HyperRam at 456544..456639 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481216 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456544 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S47_Infos, size 13 from HyperFlash at 484832 to (size 13) HyperRam at 460136..460148 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484832 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460136 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9fc268aa, size 864 from HyperFlash at 450672 to (size 864) HyperRam at 426144..427007 */
	{
		int Size = 864, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 450672 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 426144 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_11d2bf00, size 384 from HyperFlash at 467728 to (size 384) HyperRam at 443200..443583 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 467728 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443200 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S50_Mul_scale, size 96 from HyperFlash at 481312 to (size 96) HyperRam at 456640..456735 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481312 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456640 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S50_Mul_shift, size 96 from HyperFlash at 481408 to (size 96) HyperRam at 456736..456831 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481408 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456736 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S50_Infos, size 13 from HyperFlash at 484848 to (size 13) HyperRam at 460152..460164 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484848 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460152 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_f53b6b4d, size 64 from HyperFlash at 482848 to (size 64) HyperRam at 458176..458239 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482848 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458176 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S53_Mul_scale, size 16 from HyperFlash at 484864 to (size 16) HyperRam at 460168..460183 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484864 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460168 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S53_Mul_shift, size 16 from HyperFlash at 484880 to (size 16) HyperRam at 460184..460199 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484880 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460184 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S53_Infos, size 13 from HyperFlash at 484896 to (size 13) HyperRam at 460200..460212 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484896 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460200 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S54_Infos, size 13 from HyperFlash at 484912 to (size 13) HyperRam at 460216..460228 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484912 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460216 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_c68d5f56, size 384 from HyperFlash at 468112 to (size 384) HyperRam at 443584..443967 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 468112 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443584 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S57_Mul_scale, size 96 from HyperFlash at 481504 to (size 96) HyperRam at 456832..456927 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481504 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456832 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S57_Mul_shift, size 96 from HyperFlash at 481600 to (size 96) HyperRam at 456928..457023 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481600 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456928 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S57_Infos, size 13 from HyperFlash at 484928 to (size 13) HyperRam at 460232..460244 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484928 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460232 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_886f09a8, size 864 from HyperFlash at 451536 to (size 864) HyperRam at 427008..427871 */
	{
		int Size = 864, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 451536 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 427008 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_dda18fbb, size 384 from HyperFlash at 468496 to (size 384) HyperRam at 443968..444351 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 468496 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443968 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S60_Mul_scale, size 96 from HyperFlash at 481696 to (size 96) HyperRam at 457024..457119 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481696 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457024 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S60_Mul_shift, size 96 from HyperFlash at 481792 to (size 96) HyperRam at 457120..457215 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481792 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457120 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S60_Infos, size 13 from HyperFlash at 484944 to (size 13) HyperRam at 460248..460260 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484944 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460248 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_41e426e4, size 64 from HyperFlash at 482912 to (size 64) HyperRam at 458240..458303 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482912 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458240 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S63_Mul_scale, size 16 from HyperFlash at 484960 to (size 16) HyperRam at 460264..460279 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484960 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460264 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S63_Mul_shift, size 16 from HyperFlash at 484976 to (size 16) HyperRam at 460280..460295 */
	{
		int Size = 16, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484976 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460280 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S63_Infos, size 13 from HyperFlash at 484992 to (size 13) HyperRam at 460296..460308 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484992 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460296 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S64_Infos, size 13 from HyperFlash at 485008 to (size 13) HyperRam at 460312..460324 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485008 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460312 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_cabdb5c1, size 384 from HyperFlash at 468880 to (size 384) HyperRam at 444352..444735 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 468880 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 444352 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S67_Mul_scale, size 96 from HyperFlash at 481888 to (size 96) HyperRam at 457216..457311 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481888 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457216 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S67_Mul_shift, size 96 from HyperFlash at 481984 to (size 96) HyperRam at 457312..457407 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481984 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457312 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S67_Infos, size 13 from HyperFlash at 485024 to (size 13) HyperRam at 460328..460340 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485024 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460328 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_daa68995, size 864 from HyperFlash at 452400 to (size 864) HyperRam at 427872..428735 */
	{
		int Size = 864, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 452400 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 427872 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_2cc947d0, size 384 from HyperFlash at 469264 to (size 384) HyperRam at 444736..445119 */
	{
		int Size = 384, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 469264 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 444736 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S70_Mul_scale, size 96 from HyperFlash at 482080 to (size 96) HyperRam at 457408..457503 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482080 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457408 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S70_Mul_shift, size 96 from HyperFlash at 482176 to (size 96) HyperRam at 457504..457599 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482176 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457504 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S70_Infos, size 13 from HyperFlash at 485040 to (size 13) HyperRam at 460344..460356 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485040 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460344 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_bb7bc9fe, size 96 from HyperFlash at 482272 to (size 96) HyperRam at 457600..457695 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482272 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457600 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S73_Infos, size 13 from HyperFlash at 485056 to (size 13) HyperRam at 460360..460372 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485056 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460360 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_b16f9f80, size 3456 from HyperFlash at 383872 to (size 3456) HyperRam at 383872..387327 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 383872 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 383872 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_64f865bc, size 576 from HyperFlash at 458640 to (size 576) HyperRam at 434112..434687 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 458640 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 434112 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S76_Mul_scale, size 144 from HyperFlash at 477952 to (size 144) HyperRam at 453280..453423 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477952 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453280 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S76_Mul_shift, size 144 from HyperFlash at 478096 to (size 144) HyperRam at 453424..453567 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478096 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453424 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S76_Infos, size 13 from HyperFlash at 485072 to (size 13) HyperRam at 460376..460388 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485072 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460376 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_81f4a85a, size 576 from HyperFlash at 459216 to (size 576) HyperRam at 434688..435263 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 459216 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 434688 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S79_Mul_scale, size 144 from HyperFlash at 478240 to (size 144) HyperRam at 453568..453711 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478240 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453568 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S79_Mul_shift, size 144 from HyperFlash at 478384 to (size 144) HyperRam at 453712..453855 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478384 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453712 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S79_Infos, size 13 from HyperFlash at 485088 to (size 13) HyperRam at 460392..460404 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485088 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460392 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_5a4b36c9, size 3456 from HyperFlash at 387328 to (size 3456) HyperRam at 387328..390783 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 387328 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 387328 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_2764c46f, size 96 from HyperFlash at 482368 to (size 96) HyperRam at 457696..457791 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482368 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457696 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S82_Mul_scale, size 24 from HyperFlash at 484352 to (size 24) HyperRam at 459656..459679 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484352 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459656 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S82_Mul_shift, size 24 from HyperFlash at 484376 to (size 24) HyperRam at 459680..459703 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484376 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459680 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S82_Infos, size 13 from HyperFlash at 485104 to (size 13) HyperRam at 460408..460420 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485104 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460408 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S83_Infos, size 13 from HyperFlash at 485120 to (size 13) HyperRam at 460424..460436 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485120 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460424 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_3007959b, size 3456 from HyperFlash at 390784 to (size 3456) HyperRam at 390784..394239 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 390784 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 390784 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_5f1e7195, size 576 from HyperFlash at 459792 to (size 576) HyperRam at 435264..435839 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 459792 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 435264 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S86_Mul_scale, size 144 from HyperFlash at 478528 to (size 144) HyperRam at 453856..453999 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478528 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453856 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S86_Mul_shift, size 144 from HyperFlash at 478672 to (size 144) HyperRam at 454000..454143 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478672 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454000 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S86_Infos, size 13 from HyperFlash at 485136 to (size 13) HyperRam at 460440..460452 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485136 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460440 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_cc4b30bd, size 1296 from HyperFlash at 441664 to (size 1296) HyperRam at 417136..418431 */
	{
		int Size = 1296, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 441664 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 417136 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_83a3dc94, size 576 from HyperFlash at 460368 to (size 576) HyperRam at 435840..436415 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 460368 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 435840 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S89_Mul_scale, size 144 from HyperFlash at 478816 to (size 144) HyperRam at 454144..454287 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478816 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454144 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S89_Mul_shift, size 144 from HyperFlash at 478960 to (size 144) HyperRam at 454288..454431 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 478960 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454288 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S89_Infos, size 13 from HyperFlash at 485152 to (size 13) HyperRam at 460456..460468 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485152 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460456 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9a51d69c, size 3456 from HyperFlash at 394240 to (size 3456) HyperRam at 394240..397695 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 394240 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 394240 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_25106508, size 96 from HyperFlash at 482464 to (size 96) HyperRam at 457792..457887 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482464 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457792 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S92_Mul_scale, size 24 from HyperFlash at 484400 to (size 24) HyperRam at 459704..459727 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484400 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459704 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S92_Mul_shift, size 24 from HyperFlash at 484424 to (size 24) HyperRam at 459728..459751 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484424 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459728 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S92_Infos, size 13 from HyperFlash at 485168 to (size 13) HyperRam at 460472..460484 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485168 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460472 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S93_Infos, size 13 from HyperFlash at 485184 to (size 13) HyperRam at 460488..460500 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485184 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460488 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9844f6db, size 3456 from HyperFlash at 397696 to (size 3456) HyperRam at 397696..401151 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 397696 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 397696 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_75209268, size 576 from HyperFlash at 460944 to (size 576) HyperRam at 436416..436991 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 460944 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 436416 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S96_Mul_scale, size 144 from HyperFlash at 479104 to (size 144) HyperRam at 454432..454575 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479104 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454432 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S96_Mul_shift, size 144 from HyperFlash at 479248 to (size 144) HyperRam at 454576..454719 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479248 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454576 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S96_Infos, size 13 from HyperFlash at 485200 to (size 13) HyperRam at 460504..460516 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485200 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460504 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_8f1ecd3b, size 1296 from HyperFlash at 442960 to (size 1296) HyperRam at 418432..419727 */
	{
		int Size = 1296, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 442960 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 418432 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_69b825c6, size 576 from HyperFlash at 461520 to (size 576) HyperRam at 436992..437567 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 461520 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 436992 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S99_Mul_scale, size 144 from HyperFlash at 479392 to (size 144) HyperRam at 454720..454863 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479392 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454720 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S99_Mul_shift, size 144 from HyperFlash at 479536 to (size 144) HyperRam at 454864..455007 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479536 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454864 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S99_Infos, size 13 from HyperFlash at 485216 to (size 13) HyperRam at 460520..460532 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485216 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460520 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_266b0c79, size 3456 from HyperFlash at 401152 to (size 3456) HyperRam at 401152..404607 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 401152 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 401152 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_84adc68c, size 96 from HyperFlash at 482560 to (size 96) HyperRam at 457888..457983 */
	{
		int Size = 96, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482560 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457888 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S102_Mul_scale, size 24 from HyperFlash at 484448 to (size 24) HyperRam at 459752..459775 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484448 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459752 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S102_Mul_shift, size 24 from HyperFlash at 484472 to (size 24) HyperRam at 459776..459799 */
	{
		int Size = 24, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484472 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459776 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S102_Infos, size 13 from HyperFlash at 485232 to (size 13) HyperRam at 460536..460548 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485232 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460536 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S103_Infos, size 13 from HyperFlash at 485248 to (size 13) HyperRam at 460552..460564 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485248 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460552 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_a9d3a918, size 3456 from HyperFlash at 404608 to (size 3456) HyperRam at 404608..408063 */
	{
		int Size = 3456, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 404608 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 404608 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_4e5c1323, size 576 from HyperFlash at 462096 to (size 576) HyperRam at 437568..438143 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 462096 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 437568 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S106_Mul_scale, size 144 from HyperFlash at 479680 to (size 144) HyperRam at 455008..455151 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479680 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455008 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S106_Mul_shift, size 144 from HyperFlash at 479824 to (size 144) HyperRam at 455152..455295 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479824 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455152 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S106_Infos, size 13 from HyperFlash at 485264 to (size 13) HyperRam at 460568..460580 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485264 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460568 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_2fc4f9a9, size 1296 from HyperFlash at 444256 to (size 1296) HyperRam at 419728..421023 */
	{
		int Size = 1296, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 444256 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 419728 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9b056684, size 576 from HyperFlash at 462672 to (size 576) HyperRam at 438144..438719 */
	{
		int Size = 576, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 462672 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 438144 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S109_Mul_scale, size 144 from HyperFlash at 479968 to (size 144) HyperRam at 455296..455439 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 479968 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455296 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S109_Mul_shift, size 144 from HyperFlash at 480112 to (size 144) HyperRam at 455440..455583 */
	{
		int Size = 144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480112 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455440 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S109_Infos, size 13 from HyperFlash at 485280 to (size 13) HyperRam at 460584..460596 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485280 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460584 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_915f853b, size 4608 from HyperFlash at 379264 to (size 4608) HyperRam at 379264..383871 */
	{
		int Size = 4608, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 379264 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 379264 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_c09767cb, size 128 from HyperFlash at 480384 to (size 128) HyperRam at 455712..455839 */
	{
		int Size = 128, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480384 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455712 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S112_Mul_scale, size 32 from HyperFlash at 484048 to (size 32) HyperRam at 459376..459407 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484048 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459376 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S112_Mul_shift, size 32 from HyperFlash at 484080 to (size 32) HyperRam at 459408..459439 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484080 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459408 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S112_Infos, size 13 from HyperFlash at 485296 to (size 13) HyperRam at 460600..460612 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485296 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460600 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_701cc74b, size 6144 from HyperFlash at 338304 to (size 6144) HyperRam at 338304..344447 */
	{
		int Size = 6144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 338304 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 338304 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_0d88a5ec, size 768 from HyperFlash at 454032 to (size 768) HyperRam at 429504..430271 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 454032 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 429504 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S115_Mul_scale, size 192 from HyperFlash at 475504 to (size 192) HyperRam at 450976..451167 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 475504 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450976 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S115_Mul_shift, size 192 from HyperFlash at 475696 to (size 192) HyperRam at 451168..451359 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 475696 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451168 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S115_Infos, size 13 from HyperFlash at 485312 to (size 13) HyperRam at 460616..460628 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485312 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460616 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_1a3894e6, size 768 from HyperFlash at 454800 to (size 768) HyperRam at 430272..431039 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 454800 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 430272 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S118_Mul_scale, size 192 from HyperFlash at 475888 to (size 192) HyperRam at 451360..451551 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 475888 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451360 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S118_Mul_shift, size 192 from HyperFlash at 476080 to (size 192) HyperRam at 451552..451743 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 476080 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451552 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S118_Infos, size 13 from HyperFlash at 485328 to (size 13) HyperRam at 460632..460644 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485328 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460632 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_ec3a71e1, size 6144 from HyperFlash at 344448 to (size 6144) HyperRam at 344448..350591 */
	{
		int Size = 6144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 344448 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 344448 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_4a922648, size 128 from HyperFlash at 480512 to (size 128) HyperRam at 455840..455967 */
	{
		int Size = 128, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480512 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455840 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S121_Mul_scale, size 32 from HyperFlash at 484112 to (size 32) HyperRam at 459440..459471 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484112 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459440 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S121_Mul_shift, size 32 from HyperFlash at 484144 to (size 32) HyperRam at 459472..459503 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484144 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459472 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S121_Infos, size 13 from HyperFlash at 485344 to (size 13) HyperRam at 460648..460660 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485344 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460648 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S122_Infos, size 13 from HyperFlash at 485360 to (size 13) HyperRam at 460664..460676 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485360 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460664 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_83535371, size 6144 from HyperFlash at 350592 to (size 6144) HyperRam at 350592..356735 */
	{
		int Size = 6144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 350592 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 350592 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_f0f41adb, size 768 from HyperFlash at 455568 to (size 768) HyperRam at 431040..431807 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 455568 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 431040 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S125_Mul_scale, size 192 from HyperFlash at 476272 to (size 192) HyperRam at 451744..451935 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 476272 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451744 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S125_Mul_shift, size 192 from HyperFlash at 476464 to (size 192) HyperRam at 451936..452127 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 476464 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451936 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S125_Infos, size 13 from HyperFlash at 485376 to (size 13) HyperRam at 460680..460692 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485376 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460680 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_e1b513dd, size 768 from HyperFlash at 456336 to (size 768) HyperRam at 431808..432575 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 456336 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 431808 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S128_Mul_scale, size 192 from HyperFlash at 476656 to (size 192) HyperRam at 452128..452319 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 476656 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452128 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S128_Mul_shift, size 192 from HyperFlash at 476848 to (size 192) HyperRam at 452320..452511 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 476848 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452320 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S128_Infos, size 13 from HyperFlash at 485392 to (size 13) HyperRam at 460696..460708 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485392 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460696 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_d6518a03, size 6144 from HyperFlash at 356736 to (size 6144) HyperRam at 356736..362879 */
	{
		int Size = 6144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 356736 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 356736 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_97ec88ff, size 128 from HyperFlash at 480640 to (size 128) HyperRam at 455968..456095 */
	{
		int Size = 128, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480640 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455968 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S131_Mul_scale, size 32 from HyperFlash at 484176 to (size 32) HyperRam at 459504..459535 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484176 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459504 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S131_Mul_shift, size 32 from HyperFlash at 484208 to (size 32) HyperRam at 459536..459567 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484208 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459536 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S131_Infos, size 13 from HyperFlash at 485408 to (size 13) HyperRam at 460712..460724 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485408 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460712 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S132_Infos, size 13 from HyperFlash at 485424 to (size 13) HyperRam at 460728..460740 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485424 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460728 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_5e6564a2, size 6144 from HyperFlash at 362880 to (size 6144) HyperRam at 362880..369023 */
	{
		int Size = 6144, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 362880 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 362880 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_413d1607, size 768 from HyperFlash at 457104 to (size 768) HyperRam at 432576..433343 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 457104 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 432576 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S135_Mul_scale, size 192 from HyperFlash at 477040 to (size 192) HyperRam at 452512..452703 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477040 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452512 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S135_Mul_shift, size 192 from HyperFlash at 477232 to (size 192) HyperRam at 452704..452895 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477232 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452704 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S135_Infos, size 13 from HyperFlash at 485440 to (size 13) HyperRam at 460744..460756 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485440 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460744 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_ce350816, size 768 from HyperFlash at 457872 to (size 768) HyperRam at 433344..434111 */
	{
		int Size = 768, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 457872 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 433344 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S138_Mul_scale, size 192 from HyperFlash at 477424 to (size 192) HyperRam at 452896..453087 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477424 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452896 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S138_Mul_shift, size 192 from HyperFlash at 477616 to (size 192) HyperRam at 453088..453279 */
	{
		int Size = 192, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477616 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453088 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S138_Infos, size 13 from HyperFlash at 485456 to (size 13) HyperRam at 460760..460772 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485456 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460760 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_9b37a226, size 10752 from HyperFlash at 327552 to (size 10752) HyperRam at 327552..338303 */
	{
		int Size = 10752, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 327552 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 327552 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S141_Mul_scale, size 56 from HyperFlash at 483040 to (size 56) HyperRam at 458368..458423 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483040 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458368 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S141_Mul_shift, size 56 from HyperFlash at 483096 to (size 56) HyperRam at 458424..458479 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483096 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458424 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S141_Infos, size 13 from HyperFlash at 485472 to (size 13) HyperRam at 460776..460788 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485472 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460776 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_666c577b, size 18816 from HyperFlash at 221952 to (size 18816) HyperRam at 221952..240767 */
	{
		int Size = 18816, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 221952 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 221952 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S144_Mul_scale, size 336 from HyperFlash at 469648 to (size 336) HyperRam at 445120..445455 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 469648 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445120 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S144_Mul_shift, size 336 from HyperFlash at 469984 to (size 336) HyperRam at 445456..445791 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 469984 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445456 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S144_Infos, size 13 from HyperFlash at 485488 to (size 13) HyperRam at 460792..460804 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485488 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460792 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_3822911a, size 3024 from HyperFlash at 408064 to (size 3024) HyperRam at 408064..411087 */
	{
		int Size = 3024, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 408064 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 408064 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S147_Mul_scale, size 336 from HyperFlash at 470320 to (size 336) HyperRam at 445792..446127 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 470320 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445792 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S147_Mul_shift, size 336 from HyperFlash at 470656 to (size 336) HyperRam at 446128..446463 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 470656 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446128 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S147_Infos, size 13 from HyperFlash at 485504 to (size 13) HyperRam at 460808..460820 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485504 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460808 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_96b6fec7, size 18816 from HyperFlash at 240768 to (size 18816) HyperRam at 240768..259583 */
	{
		int Size = 18816, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 240768 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 240768 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_b4c31155, size 224 from HyperFlash at 473904 to (size 224) HyperRam at 449376..449599 */
	{
		int Size = 224, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 473904 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449376 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S150_Mul_scale, size 56 from HyperFlash at 483152 to (size 56) HyperRam at 458480..458535 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483152 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458480 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S150_Mul_shift, size 56 from HyperFlash at 483208 to (size 56) HyperRam at 458536..458591 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483208 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458536 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S150_Infos, size 13 from HyperFlash at 485520 to (size 13) HyperRam at 460824..460836 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485520 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460824 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S151_Infos, size 13 from HyperFlash at 485536 to (size 13) HyperRam at 460840..460852 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485536 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460840 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_5073c59c, size 18816 from HyperFlash at 259584 to (size 18816) HyperRam at 259584..278399 */
	{
		int Size = 18816, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 259584 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 259584 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S154_Mul_scale, size 336 from HyperFlash at 470992 to (size 336) HyperRam at 446464..446799 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 470992 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446464 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S154_Mul_shift, size 336 from HyperFlash at 471328 to (size 336) HyperRam at 446800..447135 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 471328 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446800 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S154_Infos, size 13 from HyperFlash at 485552 to (size 13) HyperRam at 460856..460868 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485552 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460856 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_f8cfb438, size 3024 from HyperFlash at 411088 to (size 3024) HyperRam at 411088..414111 */
	{
		int Size = 3024, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 411088 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 411088 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S157_Mul_scale, size 336 from HyperFlash at 471664 to (size 336) HyperRam at 447136..447471 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 471664 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447136 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S157_Mul_shift, size 336 from HyperFlash at 472000 to (size 336) HyperRam at 447472..447807 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 472000 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447472 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S157_Infos, size 13 from HyperFlash at 485568 to (size 13) HyperRam at 460872..460884 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485568 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460872 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_11f0296b, size 18816 from HyperFlash at 278400 to (size 18816) HyperRam at 278400..297215 */
	{
		int Size = 18816, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 278400 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 278400 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_f3fc8846, size 224 from HyperFlash at 474128 to (size 224) HyperRam at 449600..449823 */
	{
		int Size = 224, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 474128 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449600 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S160_Mul_scale, size 56 from HyperFlash at 483264 to (size 56) HyperRam at 458592..458647 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483264 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458592 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S160_Mul_shift, size 56 from HyperFlash at 483320 to (size 56) HyperRam at 458648..458703 */
	{
		int Size = 56, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 483320 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458648 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S160_Infos, size 13 from HyperFlash at 485584 to (size 13) HyperRam at 460888..460900 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485584 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460888 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S161_Infos, size 13 from HyperFlash at 485600 to (size 13) HyperRam at 460904..460916 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485600 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460904 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_6b6bef1a, size 18816 from HyperFlash at 297216 to (size 18816) HyperRam at 297216..316031 */
	{
		int Size = 18816, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 297216 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 297216 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S164_Mul_scale, size 336 from HyperFlash at 472336 to (size 336) HyperRam at 447808..448143 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 472336 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447808 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S164_Mul_shift, size 336 from HyperFlash at 472672 to (size 336) HyperRam at 448144..448479 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 472672 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448144 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S164_Infos, size 13 from HyperFlash at 485616 to (size 13) HyperRam at 460920..460932 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485616 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460920 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_6f90883e, size 3024 from HyperFlash at 414112 to (size 3024) HyperRam at 414112..417135 */
	{
		int Size = 3024, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 414112 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 414112 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S167_Mul_scale, size 336 from HyperFlash at 473008 to (size 336) HyperRam at 448480..448815 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 473008 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448480 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S167_Mul_shift, size 336 from HyperFlash at 473344 to (size 336) HyperRam at 448816..449151 */
	{
		int Size = 336, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 473344 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448816 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S167_Infos, size 13 from HyperFlash at 485632 to (size 13) HyperRam at 460936..460948 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485632 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460936 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_52fbff0b, size 37632 from HyperFlash at 184320 to (size 37632) HyperRam at 184320..221951 */
	{
		int Size = 37632, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 184320 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 184320 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bl_606f6059, size 448 from HyperFlash at 463248 to (size 448) HyperRam at 438720..439167 */
	{
		int Size = 448, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 463248 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 438720 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S170_Mul_scale, size 112 from HyperFlash at 480896 to (size 112) HyperRam at 456224..456335 */
	{
		int Size = 112, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480896 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456224 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S170_Mul_shift, size 112 from HyperFlash at 481008 to (size 112) HyperRam at 456336..456447 */
	{
		int Size = 112, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 481008 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456336 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S170_Infos, size 13 from HyperFlash at 485648 to (size 13) HyperRam at 460952..460964 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485648 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460952 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96co_190efb1b, size 143360 from HyperFlash at 0 to (size 143360) HyperRam at 0..143359 */
	{
		int Size = 143360, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 0 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 0 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S173_Mul_scale, size 1280 from HyperFlash at 445552 to (size 1280) HyperRam at 421024..422303 */
	{
		int Size = 1280, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 445552 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 421024 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S173_Mul_shift, size 1280 from HyperFlash at 446832 to (size 1280) HyperRam at 422304..423583 */
	{
		int Size = 1280, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 446832 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 422304 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S173_Infos, size 13 from HyperFlash at 485664 to (size 13) HyperRam at 460968..460980 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485664 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460968 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialseparable_conv2d_1se, size 11520 from HyperFlash at 316032 to (size 11520) HyperRam at 316032..327551 */
	{
		int Size = 11520, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 316032 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 316032 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S176_Mul_scale, size 1280 from HyperFlash at 448112 to (size 1280) HyperRam at 423584..424863 */
	{
		int Size = 1280, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 448112 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 423584 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S176_Mul_shift, size 1280 from HyperFlash at 449392 to (size 1280) HyperRam at 424864..426143 */
	{
		int Size = 1280, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 449392 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 424864 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S176_Infos, size 13 from HyperFlash at 485680 to (size 13) HyperRam at 460984..460996 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485680 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460984 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Separable_conv2d_1bias, size 128 from HyperFlash at 480768 to (size 128) HyperRam at 456096..456223 */
	{
		int Size = 128, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 480768 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456096 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S179_Mul_scale, size 32 from HyperFlash at 484240 to (size 32) HyperRam at 459568..459599 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484240 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459568 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S179_Mul_shift, size 32 from HyperFlash at 484272 to (size 32) HyperRam at 459600..459631 */
	{
		int Size = 32, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484272 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459600 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S179_Infos, size 13 from HyperFlash at 485696 to (size 13) HyperRam at 461000..461012 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485696 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461000 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S180_Infos, size 13 from HyperFlash at 485712 to (size 13) HyperRam at 461016..461028 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485712 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461016 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialdensematmul, size 64 from HyperFlash at 482976 to (size 64) HyperRam at 458304..458367 */
	{
		int Size = 64, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482976 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458304 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Densebias, size 8 from HyperFlash at 485820 to (size 8) HyperRam at 461124..461131 */
	{
		int Size = 8, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485820 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461124 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S183_Mul_scale, size 2 from HyperFlash at 485856 to (size 2) HyperRam at 461160..461161 */
	{
		int Size = 2, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485856 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461160 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S183_Mul_shift, size 2 from HyperFlash at 485860 to (size 2) HyperRam at 461164..461165 */
	{
		int Size = 2, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485860 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461164 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S183_Infos, size 13 from HyperFlash at 485728 to (size 13) HyperRam at 461032..461044 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485728 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461032 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving S184_Infos, size 13 from HyperFlash at 485744 to (size 13) HyperRam at 461048..461060 */
	{
		int Size = 13, Base = 0;
		while (Size) {
			int Chunk = Min(Size, 1024);
			AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 485744 + Base), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 0, UchanHF1);
			AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
			AT_HYPERRAM_FC_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461048 + Base), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), Chunk, 1, UchanHR2);
			AT_HYPERRAM_FC_WAIT(&HyperRam, UchanHR2);
			Base += Chunk;
			Size -= Chunk;
		}
	}
	/* Moving Sequentialmobilenetv2_035_96bn, size 64 from HyperFlash at 482656 to (size 64) L2 at 76120..76183 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 482656), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 76120), 64, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96ex, size 144 from HyperFlash at 477808 to (size 144) L2 at 29648..29791 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 477808), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 29648), 144, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_c6bf04ca, size 1536 from HyperFlash at 424624 to (size 1536) L2 at 12608..14143 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 424624), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 12608), 1536, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_8db0f664, size 1536 from HyperFlash at 426160 to (size 1536) L2 at 14144..15679 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 426160), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 14144), 1536, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_f9b1c0db, size 1536 from HyperFlash at 427696 to (size 1536) L2 at 15680..17215 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 427696), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 15680), 1536, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_455dd475, size 1536 from HyperFlash at 429232 to (size 1536) L2 at 17216..18751 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 429232), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 17216), 1536, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_d639dcef, size 1536 from HyperFlash at 430768 to (size 1536) L2 at 18752..20287 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 430768), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 18752), 1536, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_000ffe14, size 2304 from HyperFlash at 417136 to (size 2304) L2 at 5120..7423 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 417136), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 5120), 2304, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S73_Mul_scale, size 24 from HyperFlash at 484304 to (size 24) L2 at 29792..29815 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484304), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 29792), 24, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving S73_Mul_shift, size 24 from HyperFlash at 484328 to (size 24) L2 at 76184..76207 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 484328), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 76184), 24, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_7833e667, size 1296 from HyperFlash at 440368 to (size 1296) L2 at 28352..29647 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 440368), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 28352), 1296, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_1aeb4567, size 1728 from HyperFlash at 419440 to (size 1728) L2 at 7424..9151 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 419440), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 7424), 1728, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_1d4d6859, size 1728 from HyperFlash at 421168 to (size 1728) L2 at 9152..10879 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 421168), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 9152), 1728, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_0d1bfdee, size 1728 from HyperFlash at 422896 to (size 1728) L2 at 10880..12607 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 422896), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 10880), 1728, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_548058eb, size 224 from HyperFlash at 473680 to (size 224) L2 at 75896..76119 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 473680), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 75896), 224, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_51b7e479, size 1344 from HyperFlash at 432304 to (size 1344) L2 at 20288..21631 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 432304), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 20288), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_8af10a99, size 1344 from HyperFlash at 433648 to (size 1344) L2 at 21632..22975 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 433648), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 21632), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_1be2d906, size 1344 from HyperFlash at 434992 to (size 1344) L2 at 22976..24319 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 434992), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 22976), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_00b159f0, size 1344 from HyperFlash at 436336 to (size 1344) L2 at 24320..25663 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 436336), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 24320), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_24e62b87, size 1344 from HyperFlash at 437680 to (size 1344) L2 at 25664..27007 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 437680), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 25664), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96bl_e9658647, size 1344 from HyperFlash at 439024 to (size 1344) L2 at 27008..28351 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 439024), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 27008), 1344, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialmobilenetv2_035_96co_06627b5d, size 5120 from HyperFlash at 369024 to (size 5120) L2 at 70776..75895 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 369024), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 70776), 5120, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialseparable_conv2d_1se_c86b8bf3, size 5120 from HyperFlash at 374144 to (size 5120) L2 at 0..5119 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 374144), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 0), 5120, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	/* Moving Sequentialseparable_conv2d_1se_6121c020, size 40960 from HyperFlash at 143360 to (size 40960) L2 at 29816..70775 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) classification_L3_Flash + 143360), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) classification_L2_Memory + 29816), 40960, 0, UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, UchanHF1);
	return 0;
}
int classificationCNN_Destruct()

{
	AT_HYPERRAM_FREE(&HyperRam, classification_L3_Memory, 461168);
	AT_L2_FREE(0, classification_L2_Memory_Dyn, 193792);
	AT_L2_FREE(0, classification_L2_Memory, 76208);
	AT_L1_FREE(0, classification_L1_Memory, 46696);
	AT_HYPERRAM_CLOSE(&HyperRam);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int classificationCNN_Memory(int Which)

{
	switch (Which) {
		case 0: return 46696;
		case 1: return 193792;
		case 2: return 76208;
	}
	return 0;
}
int classificationCNN(
		signed char * __restrict__ Input_1,
		signed short * __restrict__ Output_1)

{
	AT_HYPERRAM_CL_EVENT _UchanHR0, *UchanHR0 = &_UchanHR0;
	AT_HYPERRAM_CL_EVENT _UchanHR1, *UchanHR1 = &_UchanHR1;
	AT_HYPERRAM_CL_EVENT _UchanHR2, *UchanHR2 = &_UchanHR2;
	AT_HYPERRAM_CL_EVENT _UchanHR3, *UchanHR3 = &_UchanHR3;
	AT_HYPERRAM_CL_EVENT _UchanHR4, *UchanHR4 = &_UchanHR4;
	AT_HYPERRAM_CL_EVENT _UchanHR5, *UchanHR5 = &_UchanHR5;
	AT_HYPERRAM_CL_EVENT _UchanHR6, *UchanHR6 = &_UchanHR6;
	AT_HYPERRAM_CL_EVENT _UchanHR7, *UchanHR7 = &_UchanHR7;
	AT_HYPERRAM_CL_EVENT _UchanHR8, *UchanHR8 = &_UchanHR8;
	AT_HYPERRAM_CL_EVENT _UchanHR9, *UchanHR9 = &_UchanHR9;
	AT_HYPERRAM_CL_EVENT _UchanHR10, *UchanHR10 = &_UchanHR10;
	AT_HYPERRAM_CL_EVENT _UchanHR11, *UchanHR11 = &_UchanHR11;
	AT_HYPERRAM_CL_EVENT _UchanHR12, *UchanHR12 = &_UchanHR12;
	AT_HYPERRAM_CL_EVENT _UchanHR13, *UchanHR13 = &_UchanHR13;
	AT_HYPERRAM_CL_EVENT _UchanHR14, *UchanHR14 = &_UchanHR14;
	AT_HYPERRAM_CL_EVENT _UchanHR15, *UchanHR15 = &_UchanHR15;
	AT_HYPERRAM_CL_EVENT _UchanHR16, *UchanHR16 = &_UchanHR16;
	AT_HYPERRAM_CL_EVENT _UchanHR17, *UchanHR17 = &_UchanHR17;
	AT_HYPERRAM_CL_EVENT _UchanHR18, *UchanHR18 = &_UchanHR18;
	AT_HYPERRAM_CL_EVENT _UchanHR19, *UchanHR19 = &_UchanHR19;
	AT_HYPERRAM_CL_EVENT _UchanHR20, *UchanHR20 = &_UchanHR20;
	AT_HYPERRAM_CL_EVENT _UchanHR21, *UchanHR21 = &_UchanHR21;
	AT_HYPERRAM_CL_EVENT _UchanHR22, *UchanHR22 = &_UchanHR22;
	/* Moving Separable_conv2ddepthwise_kern, size 1 from HyperRam at 461132 to (size 1) L2 at 96016 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461132), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19808), 1, 0, UchanHR0);
	/* Moving Sequentialseparable_conv2dsepa, size 4 from HyperRam at 461136 to (size 4) L2 at 96020 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461136), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19812), 4, 0, UchanHR1);
	/* Moving S3_Mul_scale, size 1 from HyperRam at 461140 to (size 1) L2 at 96024 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461140), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19816), 1, 0, UchanHR2);
	/* Moving S3_Mul_shift, size 1 from HyperRam at 461144 to (size 1) L2 at 96028 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461144), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19820), 1, 0, UchanHR3);
	/* Moving S3_Infos, size 13 from HyperRam at 459800 to (size 13) L2 at 95972 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459800), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19764), 13, 0, UchanHR4);
	/* Moving Sequentialseparable_conv2dsepa_411470e1, size 3 from HyperRam at 461148 to (size 3) L2 at 96032 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461148), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19824), 3, 0, UchanHR5);
	/* Moving Sequentialseparable_conv2dbias, size 12 from HyperRam at 461064 to (size 12) L2 at 96004 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461064), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19796), 12, 0, UchanHR6);
	/* Moving S6_Mul_scale, size 3 from HyperRam at 461152 to (size 3) L2 at 96036 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461152), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19828), 3, 0, UchanHR7);
	/* Moving S6_Mul_shift, size 3 from HyperRam at 461156 to (size 3) L2 at 96040 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461156), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19832), 3, 0, UchanHR8);
	/* Moving S6_Infos, size 13 from HyperRam at 459816 to (size 13) L2 at 95988 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459816), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19780), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96co, size 432 from HyperRam at 439168 to (size 432) L2 at 163148 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 439168), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 86940), 432, 0, UchanHR10);
	/* Moving S10_Mul_scale, size 16 from HyperRam at 459832 to (size 16) L2 at 163644 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459832), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 87436), 16, 0, UchanHR11);
	/* Moving S10_Mul_shift, size 16 from HyperRam at 459848 to (size 16) L2 at 163660 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459848), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 87452), 16, 0, UchanHR12);
	/* Moving S10_Infos, size 13 from HyperRam at 459864 to (size 13) L2 at 163676 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459864), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 87468), 13, 0, UchanHR13);
	/* Waiting completion of transfer of Separable_conv2ddepthwise_kern using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialseparable_conv2dsepa using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S3_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S3_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S3_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S3_Conv2d_1x1x1x1(
		((signed char * __restrict__) Input_1), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19808)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+19812)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+19816)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19820)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19764)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_ecf73e9c, size 384 from HyperRam at 440896 to (size 384) L2 at 233312 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440896), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157104), 384, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_d7b8fcc3, size 192 from HyperRam at 449824 to (size 192) L2 at 233696 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449824), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157488), 192, 0, UchanHR1);
	/* Moving S19_Mul_scale, size 48 from HyperRam at 458704 to (size 48) L2 at 234080 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458704), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157872), 48, 0, UchanHR2);
	/* Moving S19_Mul_shift, size 48 from HyperRam at 458752 to (size 48) L2 at 234128 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458752), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157920), 48, 0, UchanHR3);
	/* Moving S19_Infos, size 13 from HyperRam at 459944 to (size 13) L2 at 234272 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459944), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 158064), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialseparable_conv2dsepa_411470e1 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialseparable_conv2dbias using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S6_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S6_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S6_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S6_Conv2d_3x1x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19824)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+19796)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+19828)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19832)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19780)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_5aec7246, size 432 from HyperRam at 439600 to (size 432) L2 at 232880 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 439600), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 156672), 432, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_96627e8b, size 192 from HyperRam at 450016 to (size 192) L2 at 233888 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450016), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157680), 192, 0, UchanHR6);
	/* Moving S22_Mul_scale, size 48 from HyperRam at 458800 to (size 48) L2 at 234176 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458800), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157968), 48, 0, UchanHR7);
	/* Moving S22_Mul_shift, size 48 from HyperRam at 458848 to (size 48) L2 at 234224 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458848), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 158016), 48, 0, UchanHR8);
	/* Moving S22_Infos, size 13 from HyperRam at 459960 to (size 13) L2 at 234288 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459960), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 158080), 13, 0, UchanHR9);
	S7_Op_RESIZE_BILINEAR_0_3(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)) /* Out */
	);
	/* Moving Sequentialmobilenetv2_035_96ex_2e8d685a, size 64 from HyperRam at 458048 to (size 64) L2 at 150064 using event 14 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458048), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73856), 64, 0, UchanHR14);
	/* Moving S13_Mul_scale, size 16 from HyperRam at 459880 to (size 16) L2 at 150160 using event 15 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459880), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73952), 16, 0, UchanHR15);
	/* Moving S13_Mul_shift, size 16 from HyperRam at 459896 to (size 16) L2 at 150176 using event 16 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459896), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73968), 16, 0, UchanHR16);
	/* Moving S13_Infos, size 13 from HyperRam at 459912 to (size 13) L2 at 150192 using event 17 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459912), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73984), 13, 0, UchanHR17);
	/* Moving Sequentialmobilenetv2_035_96ex_4dd58ac9, size 128 from HyperRam at 455584 to (size 128) L2 at 149936 using event 18 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455584), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73728), 128, 0, UchanHR18);
	/* Moving Sequentialmobilenetv2_035_96ex_ec77a7bf, size 32 from HyperRam at 459280 to (size 32) L2 at 150128 using event 19 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459280), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 73920), 32, 0, UchanHR19);
	/* Moving S16_Mul_scale, size 8 from HyperRam at 461076 to (size 8) L2 at 150224 using event 20 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461076), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 74016), 8, 0, UchanHR20);
	/* Moving S16_Mul_shift, size 8 from HyperRam at 461084 to (size 8) L2 at 150232 using event 21 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461084), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 74024), 8, 0, UchanHR21);
	/* Moving S16_Infos, size 13 from HyperRam at 459928 to (size 13) L2 at 150208 using event 22 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459928), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 74000), 13, 0, UchanHR22);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96co using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S10_Mul_scale using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S10_Mul_shift using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S10_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S10_Conv2d_16x3x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+86940)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+76120)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+36864)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+87436)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+87452)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+87468)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96ex_2e8d685a using event 14 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR14);
	/* Waiting completion of transfer of S13_Mul_scale using event 15 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR15);
	/* Waiting completion of transfer of S13_Mul_shift using event 16 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR16);
	/* Waiting completion of transfer of S13_Infos using event 17 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR17);
	S13_Conv2d_16x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+36864)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+29648)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+73856)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+73952)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+73968)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+73984)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96ex_4dd58ac9 using event 18 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR18);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96ex_ec77a7bf using event 19 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR19);
	/* Waiting completion of transfer of S16_Mul_scale using event 20 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR20);
	/* Waiting completion of transfer of S16_Mul_shift using event 21 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR21);
	/* Waiting completion of transfer of S16_Infos using event 22 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR22);
	S16_Conv2d_8x16x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+73728)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+73920)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+138240)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+74016)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+74024)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+74000)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_ecf73e9c using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_d7b8fcc3 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S19_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S19_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S19_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S19_Conv2d_48x8x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+138240)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+157104)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+157488)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+157872)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+157920)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+158064)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_cb056cce, size 384 from HyperRam at 441280 to (size 384) L2 at 214448 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 441280), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 138240), 384, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_adb8c218, size 32 from HyperRam at 459312 to (size 32) L2 at 214832 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459312), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 138624), 32, 0, UchanHR1);
	/* Moving S25_Mul_scale, size 8 from HyperRam at 461092 to (size 8) L2 at 214880 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461092), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 138672), 8, 0, UchanHR2);
	/* Moving S25_Mul_shift, size 8 from HyperRam at 461100 to (size 8) L2 at 214888 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461100), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 138680), 8, 0, UchanHR3);
	/* Moving S25_Infos, size 13 from HyperRam at 459976 to (size 13) L2 at 214864 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459976), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 138656), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_5aec7246 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_96627e8b using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S22_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S22_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S22_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S22_Conv2d_48x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+156672)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+157680)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+157968)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+158016)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+158080)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_fc011502, size 384 from HyperRam at 441664 to (size 384) L2 at 108464 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 441664), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 32256), 384, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_2be52376, size 192 from HyperRam at 450208 to (size 192) L2 at 108848 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450208), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 32640), 192, 0, UchanHR6);
	/* Moving S28_Mul_scale, size 48 from HyperRam at 458896 to (size 48) L2 at 109040 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458896), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 32832), 48, 0, UchanHR7);
	/* Moving S28_Mul_shift, size 48 from HyperRam at 458944 to (size 48) L2 at 109088 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458944), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 32880), 48, 0, UchanHR8);
	/* Moving S28_Infos, size 13 from HyperRam at 459992 to (size 13) L2 at 109136 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459992), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 32928), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_4d2ce9de, size 432 from HyperRam at 440032 to (size 432) L2 at 136112 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440032), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 59904), 432, 0, UchanHR10);
	/* Moving Sequentialmobilenetv2_035_96bl_2448ea8a, size 192 from HyperRam at 450400 to (size 192) L2 at 136928 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450400), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 60720), 192, 0, UchanHR11);
	/* Moving S31_Mul_scale, size 48 from HyperRam at 458992 to (size 48) L2 at 137120 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458992), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 60912), 48, 0, UchanHR12);
	/* Moving S31_Mul_shift, size 48 from HyperRam at 459040 to (size 48) L2 at 137168 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459040), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 60960), 48, 0, UchanHR13);
	/* Moving S31_Infos, size 13 from HyperRam at 460008 to (size 13) L2 at 137248 using event 14 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460008), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 61040), 13, 0, UchanHR14);
	/* Moving Sequentialmobilenetv2_035_96bl_9757a219, size 384 from HyperRam at 442048 to (size 384) L2 at 136544 using event 15 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442048), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 60336), 384, 0, UchanHR15);
	/* Moving Sequentialmobilenetv2_035_96bl_fc9e5e40, size 32 from HyperRam at 459344 to (size 32) L2 at 137216 using event 16 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459344), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 61008), 32, 0, UchanHR16);
	/* Moving S34_Mul_scale, size 8 from HyperRam at 461108 to (size 8) L2 at 137280 using event 17 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461108), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 61072), 8, 0, UchanHR17);
	/* Moving S34_Mul_shift, size 8 from HyperRam at 461116 to (size 8) L2 at 137288 using event 18 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461116), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 61080), 8, 0, UchanHR18);
	/* Moving S34_Infos, size 13 from HyperRam at 460024 to (size 13) L2 at 137264 using event 19 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460024), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 61056), 13, 0, UchanHR19);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_cb056cce using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_adb8c218 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S25_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S25_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S25_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S25_Conv2d_8x48x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+138240)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+138624)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+138672)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+138680)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+138656)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_fc011502 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_2be52376 using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S28_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S28_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S28_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S28_Conv2d_48x8x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32256)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+32640)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+32832)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32880)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32928)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_4d2ce9de using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_2448ea8a using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S31_Mul_scale using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S31_Mul_shift using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	/* Waiting completion of transfer of S31_Infos using event 14 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR14);
	S31_Conv2d_48x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+59904)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+60720)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32256)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+60912)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+60960)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+61040)) /* Infos */
	);
	/* Moving S35_Infos, size 13 from HyperRam at 460040 to (size 13) L2 at 80816 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460040), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 4608), 13, 0, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9757a219 using event 15 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR15);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_fc9e5e40 using event 16 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR16);
	/* Waiting completion of transfer of S34_Mul_scale using event 17 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR17);
	/* Waiting completion of transfer of S34_Mul_shift using event 18 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR18);
	/* Waiting completion of transfer of S34_Infos using event 19 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR19);
	S34_Conv2d_8x48x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32256)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+60336)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+61008)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+61072)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+61080)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+61056)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_4b74c721, size 384 from HyperRam at 442432 to (size 384) L2 at 113504 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442432), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 37296), 384, 0, UchanHR1);
	/* Moving Sequentialmobilenetv2_035_96bl_9be0ce95, size 192 from HyperRam at 450592 to (size 192) L2 at 113888 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450592), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 37680), 192, 0, UchanHR2);
	/* Moving S38_Mul_scale, size 48 from HyperRam at 459088 to (size 48) L2 at 114272 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459088), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38064), 48, 0, UchanHR3);
	/* Moving S38_Mul_shift, size 48 from HyperRam at 459136 to (size 48) L2 at 114320 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459136), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38112), 48, 0, UchanHR4);
	/* Moving S38_Infos, size 13 from HyperRam at 460056 to (size 13) L2 at 114464 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460056), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38256), 13, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_d944b225, size 432 from HyperRam at 440464 to (size 432) L2 at 113072 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 440464), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 36864), 432, 0, UchanHR6);
	/* Moving Sequentialmobilenetv2_035_96bl_22a2b491, size 192 from HyperRam at 450784 to (size 192) L2 at 114080 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450784), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 37872), 192, 0, UchanHR7);
	/* Moving S41_Mul_scale, size 48 from HyperRam at 459184 to (size 48) L2 at 114368 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459184), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38160), 48, 0, UchanHR8);
	/* Moving S41_Mul_shift, size 48 from HyperRam at 459232 to (size 48) L2 at 114416 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459232), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38208), 48, 0, UchanHR9);
	/* Moving S41_Infos, size 13 from HyperRam at 460072 to (size 13) L2 at 114480 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460072), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38272), 13, 0, UchanHR10);
	/* Waiting completion of transfer of S35_Infos using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	S35_MatAdd_8x24x24(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32256)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+4608)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_4b74c721 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9be0ce95 using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S38_Mul_scale using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S38_Mul_shift using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S38_Infos using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	S38_Conv2d_48x8x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32256)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+37296)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+37680)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+38064)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38112)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38256)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_b7f5ec57, size 768 from HyperRam at 428736 to (size 768) L2 at 110768 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 428736), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 34560), 768, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_1f34f003, size 64 from HyperRam at 458112 to (size 64) L2 at 111536 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458112), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 35328), 64, 0, UchanHR1);
	/* Moving S44_Mul_scale, size 16 from HyperRam at 460088 to (size 16) L2 at 111600 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460088), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 35392), 16, 0, UchanHR2);
	/* Moving S44_Mul_shift, size 16 from HyperRam at 460104 to (size 16) L2 at 111616 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460104), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 35408), 16, 0, UchanHR3);
	/* Moving S44_Infos, size 13 from HyperRam at 460120 to (size 13) L2 at 111632 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460120), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 35424), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_d944b225 using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_22a2b491 using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S41_Mul_scale using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S41_Mul_shift using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S41_Infos using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	S41_Conv2d_48x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+36864)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+37872)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+38160)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38208)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38272)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_6437f573, size 384 from HyperRam at 442816 to (size 384) L2 at 92336 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 442816), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16128), 384, 0, UchanHR5);
	/* Moving S47_Mul_scale, size 96 from HyperRam at 456448 to (size 96) L2 at 92720 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456448), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16512), 96, 0, UchanHR6);
	/* Moving S47_Mul_shift, size 96 from HyperRam at 456544 to (size 96) L2 at 92816 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456544), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16608), 96, 0, UchanHR7);
	/* Moving S47_Infos, size 13 from HyperRam at 460136 to (size 13) L2 at 92912 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460136), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16704), 13, 0, UchanHR8);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_b7f5ec57 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_1f34f003 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S44_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S44_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S44_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S44_Conv2d_16x48x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+34560)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+35328)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+35392)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+35408)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+35424)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_9fc268aa, size 864 from HyperRam at 426144 to (size 864) L2 at 106160 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 426144), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29952), 864, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_11d2bf00, size 384 from HyperRam at 443200 to (size 384) L2 at 107024 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443200), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 30816), 384, 0, UchanHR1);
	/* Moving S50_Mul_scale, size 96 from HyperRam at 456640 to (size 96) L2 at 107408 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456640), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31200), 96, 0, UchanHR2);
	/* Moving S50_Mul_shift, size 96 from HyperRam at 456736 to (size 96) L2 at 107504 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456736), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31296), 96, 0, UchanHR3);
	/* Moving S50_Infos, size 13 from HyperRam at 460152 to (size 13) L2 at 107664 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460152), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31456), 13, 0, UchanHR4);
	/* Moving Sequentialmobilenetv2_035_96bl_f53b6b4d, size 64 from HyperRam at 458176 to (size 64) L2 at 107600 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458176), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31392), 64, 0, UchanHR9);
	/* Moving S53_Mul_scale, size 16 from HyperRam at 460168 to (size 16) L2 at 107680 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460168), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31472), 16, 0, UchanHR10);
	/* Moving S53_Mul_shift, size 16 from HyperRam at 460184 to (size 16) L2 at 107696 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460184), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31488), 16, 0, UchanHR11);
	/* Moving S53_Infos, size 13 from HyperRam at 460200 to (size 13) L2 at 107712 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460200), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31504), 13, 0, UchanHR12);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_6437f573 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S47_Mul_scale using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S47_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S47_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S47_Conv2d_96x16x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+12608)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+16128)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+2304)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+16512)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16608)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16704)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9fc268aa using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_11d2bf00 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S50_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S50_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S50_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S50_Conv2d_96x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+2304)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29952)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+30816)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16128)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+31200)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31296)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31456)) /* Infos */
	);
	/* Moving S54_Infos, size 13 from HyperRam at 460216 to (size 13) L2 at 80816 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460216), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 4608), 13, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_c68d5f56, size 384 from HyperRam at 443584 to (size 384) L2 at 90032 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443584), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13824), 384, 0, UchanHR1);
	/* Moving S57_Mul_scale, size 96 from HyperRam at 456832 to (size 96) L2 at 90416 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456832), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14208), 96, 0, UchanHR2);
	/* Moving S57_Mul_shift, size 96 from HyperRam at 456928 to (size 96) L2 at 90512 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456928), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14304), 96, 0, UchanHR3);
	/* Moving S57_Infos, size 13 from HyperRam at 460232 to (size 13) L2 at 90608 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460232), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14400), 13, 0, UchanHR4);
	/* Moving Sequentialmobilenetv2_035_96bl_886f09a8, size 864 from HyperRam at 427008 to (size 864) L2 at 106160 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 427008), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29952), 864, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_dda18fbb, size 384 from HyperRam at 443968 to (size 384) L2 at 107024 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 443968), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 30816), 384, 0, UchanHR6);
	/* Moving S60_Mul_scale, size 96 from HyperRam at 457024 to (size 96) L2 at 107408 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457024), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31200), 96, 0, UchanHR7);
	/* Moving S60_Mul_shift, size 96 from HyperRam at 457120 to (size 96) L2 at 107504 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457120), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31296), 96, 0, UchanHR8);
	/* Moving S60_Infos, size 13 from HyperRam at 460248 to (size 13) L2 at 107664 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460248), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31456), 13, 0, UchanHR13);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_f53b6b4d using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S53_Mul_scale using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S53_Mul_shift using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S53_Infos using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	S53_Conv2d_16x96x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16128)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+14144)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+31392)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+2304)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+31472)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31488)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31504)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_41e426e4, size 64 from HyperRam at 458240 to (size 64) L2 at 107600 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458240), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31392), 64, 0, UchanHR9);
	/* Moving S63_Mul_scale, size 16 from HyperRam at 460264 to (size 16) L2 at 107680 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460264), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31472), 16, 0, UchanHR10);
	/* Moving S63_Mul_shift, size 16 from HyperRam at 460280 to (size 16) L2 at 107696 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460280), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31488), 16, 0, UchanHR11);
	/* Moving S63_Infos, size 13 from HyperRam at 460296 to (size 13) L2 at 107712 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460296), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31504), 13, 0, UchanHR12);
	/* Waiting completion of transfer of S54_Infos using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	S54_MatAdd_16x12x12(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+2304)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+4608)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_c68d5f56 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S57_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S57_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S57_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S57_Conv2d_96x16x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+15680)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+13824)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+14208)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+14304)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+14400)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_886f09a8 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_dda18fbb using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S60_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S60_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S60_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S60_Conv2d_96x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29952)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+30816)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+31200)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31296)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31456)) /* Infos */
	);
	/* Moving S64_Infos, size 13 from HyperRam at 460312 to (size 13) L2 at 78512 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460312), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 2304), 13, 0, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_41e426e4 using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S63_Mul_scale using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S63_Mul_shift using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S63_Infos using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	S63_Conv2d_16x96x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+17216)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+31392)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+31472)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31488)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31504)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_cabdb5c1, size 384 from HyperRam at 444352 to (size 384) L2 at 92336 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 444352), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16128), 384, 0, UchanHR1);
	/* Moving S67_Mul_scale, size 96 from HyperRam at 457216 to (size 96) L2 at 92720 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457216), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16512), 96, 0, UchanHR2);
	/* Moving S67_Mul_shift, size 96 from HyperRam at 457312 to (size 96) L2 at 92816 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457312), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16608), 96, 0, UchanHR3);
	/* Moving S67_Infos, size 13 from HyperRam at 460328 to (size 13) L2 at 92912 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460328), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16704), 13, 0, UchanHR4);
	/* Moving Sequentialmobilenetv2_035_96bl_daa68995, size 864 from HyperRam at 427872 to (size 864) L2 at 93488 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 427872), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 17280), 864, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_2cc947d0, size 384 from HyperRam at 444736 to (size 384) L2 at 94352 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 444736), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18144), 384, 0, UchanHR6);
	/* Moving S70_Mul_scale, size 96 from HyperRam at 457408 to (size 96) L2 at 94736 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457408), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18528), 96, 0, UchanHR7);
	/* Moving S70_Mul_shift, size 96 from HyperRam at 457504 to (size 96) L2 at 94832 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457504), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18624), 96, 0, UchanHR8);
	/* Moving S70_Infos, size 13 from HyperRam at 460344 to (size 13) L2 at 95048 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460344), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18840), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_bb7bc9fe, size 96 from HyperRam at 457600 to (size 96) L2 at 94928 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457600), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18720), 96, 0, UchanHR10);
	/* Moving S73_Infos, size 13 from HyperRam at 460360 to (size 13) L2 at 95064 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460360), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 18856), 13, 0, UchanHR11);
	/* Waiting completion of transfer of S64_Infos using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	S64_MatAdd_16x12x12(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27648)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+2304)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_cabdb5c1 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S67_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S67_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S67_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S67_Conv2d_96x16x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+18752)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+16128)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+16512)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16608)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16704)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_daa68995 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_2cc947d0 using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S70_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S70_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S70_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S70_Conv2d_96x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+17280)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+18144)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+18528)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+18624)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+18840)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_b16f9f80, size 3456 from HyperRam at 383872 to (size 3456) L2 at 82256 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 383872), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6048), 3456, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_64f865bc, size 576 from HyperRam at 434112 to (size 576) L2 at 85712 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 434112), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9504), 576, 0, UchanHR1);
	/* Moving S76_Mul_scale, size 144 from HyperRam at 453280 to (size 144) L2 at 86288 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453280), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10080), 144, 0, UchanHR2);
	/* Moving S76_Mul_shift, size 144 from HyperRam at 453424 to (size 144) L2 at 86432 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453424), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10224), 144, 0, UchanHR3);
	/* Moving S76_Infos, size 13 from HyperRam at 460376 to (size 13) L2 at 86576 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460376), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10368), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_bb7bc9fe using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S73_Infos using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	S73_Conv2d_24x96x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory+5120)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+18720)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory+29792)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory+76184)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+18856)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_81f4a85a, size 576 from HyperRam at 434688 to (size 576) L2 at 90896 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 434688), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14688), 576, 0, UchanHR5);
	/* Moving S79_Mul_scale, size 144 from HyperRam at 453568 to (size 144) L2 at 91472 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453568), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15264), 144, 0, UchanHR6);
	/* Moving S79_Mul_shift, size 144 from HyperRam at 453712 to (size 144) L2 at 91616 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453712), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15408), 144, 0, UchanHR7);
	/* Moving S79_Infos, size 13 from HyperRam at 460392 to (size 13) L2 at 91904 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460392), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15696), 13, 0, UchanHR8);
	/* Moving Sequentialmobilenetv2_035_96bl_5a4b36c9, size 3456 from HyperRam at 387328 to (size 3456) L2 at 87440 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 387328), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11232), 3456, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_2764c46f, size 96 from HyperRam at 457696 to (size 96) L2 at 91760 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457696), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15552), 96, 0, UchanHR10);
	/* Moving S82_Mul_scale, size 24 from HyperRam at 459656 to (size 24) L2 at 91856 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459656), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15648), 24, 0, UchanHR11);
	/* Moving S82_Mul_shift, size 24 from HyperRam at 459680 to (size 24) L2 at 91880 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459680), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15672), 24, 0, UchanHR12);
	/* Moving S82_Infos, size 13 from HyperRam at 460408 to (size 13) L2 at 91920 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460408), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15712), 13, 0, UchanHR13);
	/* Moving Sequentialmobilenetv2_035_96bl_83a3dc94, size 576 from HyperRam at 435840 to (size 576) L2 at 92192 using event 14 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 435840), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15984), 576, 0, UchanHR14);
	/* Moving S89_Mul_scale, size 144 from HyperRam at 454144 to (size 144) L2 at 92768 using event 15 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454144), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16560), 144, 0, UchanHR15);
	/* Moving S89_Mul_shift, size 144 from HyperRam at 454288 to (size 144) L2 at 92912 using event 16 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454288), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16704), 144, 0, UchanHR16);
	/* Moving S89_Infos, size 13 from HyperRam at 460456 to (size 13) L2 at 93200 using event 17 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460456), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16992), 13, 0, UchanHR17);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_b16f9f80 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_64f865bc using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S76_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S76_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S76_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S76_Conv2d_144x24x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+9504)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+10080)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10224)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_25106508, size 96 from HyperRam at 457792 to (size 96) L2 at 93056 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457792), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16848), 96, 0, UchanHR0);
	/* Moving S92_Mul_scale, size 24 from HyperRam at 459704 to (size 24) L2 at 93152 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459704), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16944), 24, 0, UchanHR1);
	/* Moving S92_Mul_shift, size 24 from HyperRam at 459728 to (size 24) L2 at 93176 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459728), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16968), 24, 0, UchanHR2);
	/* Moving S92_Infos, size 13 from HyperRam at 460472 to (size 13) L2 at 93216 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460472), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 17008), 13, 0, UchanHR3);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_81f4a85a using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S79_Mul_scale using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S79_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S79_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S79_Conv2d_144x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+28352)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+14688)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+15264)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15408)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15696)) /* Infos */
	);
	/* Moving S83_Infos, size 13 from HyperRam at 460424 to (size 13) L2 at 77936 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460424), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1728), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_5a4b36c9 using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_2764c46f using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S82_Mul_scale using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S82_Mul_shift using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S82_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S82_Conv2d_24x144x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11232)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+15552)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+15648)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15672)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15712)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_3007959b, size 3456 from HyperRam at 390784 to (size 3456) L2 at 81392 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 390784), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 5184), 3456, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_5f1e7195, size 576 from HyperRam at 435264 to (size 576) L2 at 84848 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 435264), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 8640), 576, 0, UchanHR6);
	/* Moving S86_Mul_scale, size 144 from HyperRam at 453856 to (size 144) L2 at 85424 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453856), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9216), 144, 0, UchanHR7);
	/* Moving S86_Mul_shift, size 144 from HyperRam at 454000 to (size 144) L2 at 85568 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454000), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9360), 144, 0, UchanHR8);
	/* Moving S86_Infos, size 13 from HyperRam at 460440 to (size 13) L2 at 85712 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460440), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9504), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_cc4b30bd, size 1296 from HyperRam at 417136 to (size 1296) L2 at 90896 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 417136), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14688), 1296, 0, UchanHR10);
	/* Moving Sequentialmobilenetv2_035_96bl_9a51d69c, size 3456 from HyperRam at 394240 to (size 3456) L2 at 87440 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 394240), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11232), 3456, 0, UchanHR11);
	/* Waiting completion of transfer of S83_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S83_MatAdd_24x6x6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1728)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_3007959b using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_5f1e7195 using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S86_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S86_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S86_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S86_Conv2d_144x24x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+8640)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+9216)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+9360)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+9504)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_cc4b30bd using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_83a3dc94 using event 14 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR14);
	/* Waiting completion of transfer of S89_Mul_scale using event 15 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR15);
	/* Waiting completion of transfer of S89_Mul_shift using event 16 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR16);
	/* Waiting completion of transfer of S89_Infos using event 17 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR17);
	S89_Conv2d_144x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+14688)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+15984)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+16560)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16704)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16992)) /* Infos */
	);
	/* Moving S93_Infos, size 13 from HyperRam at 460488 to (size 13) L2 at 77072 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460488), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 864), 13, 0, UchanHR4);
	/* Moving Sequentialmobilenetv2_035_96bl_8f1ecd3b, size 1296 from HyperRam at 418432 to (size 1296) L2 at 91760 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 418432), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15552), 1296, 0, UchanHR5);
	/* Moving S99_Infos, size 13 from HyperRam at 460520 to (size 13) L2 at 93200 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460520), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16992), 13, 0, UchanHR6);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9a51d69c using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_25106508 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S92_Mul_scale using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S92_Mul_shift using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S92_Infos using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	S92_Conv2d_24x144x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11232)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+16848)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+16944)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16968)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+17008)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_9844f6db, size 3456 from HyperRam at 397696 to (size 3456) L2 at 81392 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 397696), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 5184), 3456, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_75209268, size 576 from HyperRam at 436416 to (size 576) L2 at 84848 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 436416), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 8640), 576, 0, UchanHR1);
	/* Moving S96_Mul_scale, size 144 from HyperRam at 454432 to (size 144) L2 at 85424 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454432), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9216), 144, 0, UchanHR2);
	/* Moving S96_Mul_shift, size 144 from HyperRam at 454576 to (size 144) L2 at 85568 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454576), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9360), 144, 0, UchanHR3);
	/* Moving S96_Infos, size 13 from HyperRam at 460504 to (size 13) L2 at 85712 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460504), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9504), 13, 0, UchanHR7);
	/* Moving Sequentialmobilenetv2_035_96bl_266b0c79, size 3456 from HyperRam at 401152 to (size 3456) L2 at 88304 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 401152), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12096), 3456, 0, UchanHR8);
	/* Moving Sequentialmobilenetv2_035_96bl_84adc68c, size 96 from HyperRam at 457888 to (size 96) L2 at 93056 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 457888), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16848), 96, 0, UchanHR9);
	/* Moving S102_Mul_scale, size 24 from HyperRam at 459752 to (size 24) L2 at 93152 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459752), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16944), 24, 0, UchanHR10);
	/* Moving S102_Mul_shift, size 24 from HyperRam at 459776 to (size 24) L2 at 93176 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459776), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 16968), 24, 0, UchanHR11);
	/* Moving S102_Infos, size 13 from HyperRam at 460536 to (size 13) L2 at 93216 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460536), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 17008), 13, 0, UchanHR12);
	/* Waiting completion of transfer of S93_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S93_MatAdd_24x6x6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11232)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_69b825c6, size 576 from HyperRam at 436992 to (size 576) L2 at 86576 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 436992), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10368), 576, 0, UchanHR4);
	/* Moving S99_Mul_scale, size 144 from HyperRam at 454720 to (size 144) L2 at 87152 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454720), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10944), 144, 0, UchanHR13);
	/* Moving S99_Mul_shift, size 144 from HyperRam at 454864 to (size 144) L2 at 87296 using event 14 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 454864), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11088), 144, 0, UchanHR14);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9844f6db using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_75209268 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S96_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S96_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S96_Infos using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	S96_Conv2d_144x24x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11232)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+8640)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+9216)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+9360)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+9504)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_915f853b, size 4608 from HyperRam at 379264 to (size 4608) L2 at 95600 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 379264), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 19392), 4608, 0, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_8f1ecd3b using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_69b825c6 using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S99_Mul_scale using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	/* Waiting completion of transfer of S99_Mul_shift using event 14 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR14);
	/* Waiting completion of transfer of S99_Infos using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	S99_Conv2d_144x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15552)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+10368)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+10944)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11088)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16992)) /* Infos */
	);
	/* Moving S103_Infos, size 13 from HyperRam at 460552 to (size 13) L2 at 86576 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460552), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10368), 13, 0, UchanHR1);
	/* Moving Sequentialmobilenetv2_035_96bl_a9d3a918, size 3456 from HyperRam at 404608 to (size 3456) L2 at 77936 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 404608), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1728), 3456, 0, UchanHR2);
	/* Moving Sequentialmobilenetv2_035_96bl_0d88a5ec, size 768 from HyperRam at 429504 to (size 768) L2 at 100208 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 429504), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 24000), 768, 0, UchanHR3);
	/* Moving S115_Mul_scale, size 192 from HyperRam at 450976 to (size 192) L2 at 100976 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 450976), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 24768), 192, 0, UchanHR4);
	/* Moving S115_Mul_shift, size 192 from HyperRam at 451168 to (size 192) L2 at 101168 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451168), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 24960), 192, 0, UchanHR5);
	/* Moving S115_Infos, size 13 from HyperRam at 460616 to (size 13) L2 at 101360 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460616), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25152), 13, 0, UchanHR6);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_266b0c79 using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_84adc68c using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S102_Mul_scale using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S102_Mul_shift using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S102_Infos using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	S102_Conv2d_24x144x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+12096)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+16848)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+16944)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+16968)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+17008)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_4e5c1323, size 576 from HyperRam at 437568 to (size 576) L2 at 88304 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 437568), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12096), 576, 0, UchanHR7);
	/* Moving S106_Mul_scale, size 144 from HyperRam at 455008 to (size 144) L2 at 89456 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455008), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13248), 144, 0, UchanHR8);
	/* Moving S106_Mul_shift, size 144 from HyperRam at 455152 to (size 144) L2 at 89600 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455152), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13392), 144, 0, UchanHR9);
	/* Moving S106_Infos, size 13 from HyperRam at 460568 to (size 13) L2 at 89744 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460568), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13536), 13, 0, UchanHR10);
	/* Moving Sequentialmobilenetv2_035_96bl_9b056684, size 576 from HyperRam at 438144 to (size 576) L2 at 88880 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 438144), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12672), 576, 0, UchanHR11);
	/* Moving S109_Infos, size 13 from HyperRam at 460584 to (size 13) L2 at 89760 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460584), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13552), 13, 0, UchanHR12);
	/* Waiting completion of transfer of S103_Infos using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	S103_MatAdd_24x6x6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11232)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+864)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_2fc4f9a9, size 1296 from HyperRam at 419728 to (size 1296) L2 at 86576 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 419728), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10368), 1296, 0, UchanHR1);
	/* Moving S109_Mul_scale, size 144 from HyperRam at 455296 to (size 144) L2 at 87872 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455296), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11664), 144, 0, UchanHR13);
	/* Moving S109_Mul_shift, size 144 from HyperRam at 455440 to (size 144) L2 at 88016 using event 14 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455440), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11808), 144, 0, UchanHR14);
	/* Moving Sequentialmobilenetv2_035_96bl_c09767cb, size 128 from HyperRam at 455712 to (size 128) L2 at 88160 using event 15 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455712), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 11952), 128, 0, UchanHR15);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_a9d3a918 using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_4e5c1323 using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S106_Mul_scale using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S106_Mul_shift using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S106_Infos using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	S106_Conv2d_144x24x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1728)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+12096)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+13248)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13392)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13536)) /* Infos */
	);
	/* Moving S112_Mul_scale, size 32 from HyperRam at 459376 to (size 32) L2 at 88288 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459376), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12080), 32, 0, UchanHR2);
	/* Moving S112_Mul_shift, size 32 from HyperRam at 459408 to (size 32) L2 at 88320 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459408), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12112), 32, 0, UchanHR7);
	/* Moving S112_Infos, size 13 from HyperRam at 460600 to (size 13) L2 at 88352 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460600), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12144), 13, 0, UchanHR8);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_2fc4f9a9 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9b056684 using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S109_Mul_scale using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	/* Waiting completion of transfer of S109_Mul_shift using event 14 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR14);
	/* Waiting completion of transfer of S109_Infos using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	S109_Conv2d_144x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+12672)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+11664)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+11808)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13552)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_701cc74b, size 6144 from HyperRam at 338304 to (size 6144) L2 at 89456 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 338304), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13248), 6144, 0, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_915f853b using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_c09767cb using event 15 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR15);
	/* Waiting completion of transfer of S112_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S112_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S112_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S112_Conv2d_32x144x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+19392)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+11952)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+12080)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+12112)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+12144)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_1a3894e6, size 768 from HyperRam at 430272 to (size 768) L2 at 76208 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 430272), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 0), 768, 0, UchanHR0);
	/* Moving S118_Mul_scale, size 192 from HyperRam at 451360 to (size 192) L2 at 76976 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451360), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 768), 192, 0, UchanHR2);
	/* Moving S118_Mul_shift, size 192 from HyperRam at 451552 to (size 192) L2 at 77168 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451552), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 960), 192, 0, UchanHR7);
	/* Moving S118_Infos, size 13 from HyperRam at 460632 to (size 13) L2 at 77552 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460632), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1344), 13, 0, UchanHR8);
	/* Moving Sequentialmobilenetv2_035_96bl_4a922648, size 128 from HyperRam at 455840 to (size 128) L2 at 77360 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455840), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1152), 128, 0, UchanHR9);
	/* Moving S121_Mul_scale, size 32 from HyperRam at 459440 to (size 32) L2 at 77488 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459440), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1280), 32, 0, UchanHR10);
	/* Moving S121_Mul_shift, size 32 from HyperRam at 459472 to (size 32) L2 at 77520 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459472), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1312), 32, 0, UchanHR11);
	/* Moving S121_Infos, size 13 from HyperRam at 460648 to (size 13) L2 at 77568 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460648), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1360), 13, 0, UchanHR12);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_701cc74b using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_0d88a5ec using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S115_Mul_scale using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S115_Mul_shift using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S115_Infos using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	S115_Conv2d_192x32x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13248)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+24000)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6336)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+24768)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+24960)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+25152)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_ec3a71e1, size 6144 from HyperRam at 344448 to (size 6144) L2 at 96368 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 344448), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 20160), 6144, 0, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_1a3894e6 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S118_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S118_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S118_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S118_Conv2d_192x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6336)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+7424)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+0)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13248)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+768)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+960)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1344)) /* Infos */
	);
	/* Moving S122_Infos, size 13 from HyperRam at 460664 to (size 13) L2 at 77552 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460664), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1344), 13, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_83535371, size 6144 from HyperRam at 350592 to (size 6144) L2 at 83120 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 350592), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6912), 6144, 0, UchanHR2);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_ec3a71e1 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_4a922648 using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	/* Waiting completion of transfer of S121_Mul_scale using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S121_Mul_shift using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S121_Infos using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	S121_Conv2d_32x192x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13248)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+20160)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+1152)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+1280)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1312)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1360)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_f0f41adb, size 768 from HyperRam at 431040 to (size 768) L2 at 89264 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 431040), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 13056), 768, 0, UchanHR1);
	/* Moving S125_Mul_scale, size 192 from HyperRam at 451744 to (size 192) L2 at 91184 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451744), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14976), 192, 0, UchanHR3);
	/* Moving S125_Mul_shift, size 192 from HyperRam at 451936 to (size 192) L2 at 91376 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 451936), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15168), 192, 0, UchanHR4);
	/* Moving S125_Infos, size 13 from HyperRam at 460680 to (size 13) L2 at 91568 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460680), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 15360), 13, 0, UchanHR5);
	/* Moving Sequentialmobilenetv2_035_96bl_e1b513dd, size 768 from HyperRam at 431808 to (size 768) L2 at 97328 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 431808), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 21120), 768, 0, UchanHR6);
	/* Moving S128_Mul_scale, size 192 from HyperRam at 452128 to (size 192) L2 at 98096 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452128), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 21888), 192, 0, UchanHR7);
	/* Moving S128_Mul_shift, size 192 from HyperRam at 452320 to (size 192) L2 at 98288 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452320), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22080), 192, 0, UchanHR8);
	/* Moving S128_Infos, size 13 from HyperRam at 460696 to (size 13) L2 at 98672 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460696), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22464), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_97ec88ff, size 128 from HyperRam at 455968 to (size 128) L2 at 98480 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 455968), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22272), 128, 0, UchanHR10);
	/* Moving S131_Mul_scale, size 32 from HyperRam at 459504 to (size 32) L2 at 98608 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459504), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22400), 32, 0, UchanHR11);
	/* Moving S131_Mul_shift, size 32 from HyperRam at 459536 to (size 32) L2 at 98640 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459536), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22432), 32, 0, UchanHR12);
	/* Moving S131_Infos, size 13 from HyperRam at 460712 to (size 13) L2 at 98688 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460712), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 22480), 13, 0, UchanHR13);
	/* Waiting completion of transfer of S122_Infos using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	S122_MatAdd_32x6x6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+5184)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1344)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_83535371 using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_f0f41adb using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S125_Mul_scale using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S125_Mul_shift using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S125_Infos using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	S125_Conv2d_192x32x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+13056)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+14976)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15168)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+15360)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_d6518a03, size 6144 from HyperRam at 356736 to (size 6144) L2 at 91184 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 356736), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14976), 6144, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_9b37a226, size 10752 from HyperRam at 327552 to (size 10752) L2 at 103664 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 327552), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 27456), 10752, 0, UchanHR1);
	/* Moving S141_Mul_scale, size 56 from HyperRam at 458368 to (size 56) L2 at 114640 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458368), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38432), 56, 0, UchanHR2);
	/* Moving S141_Mul_shift, size 56 from HyperRam at 458424 to (size 56) L2 at 114696 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458424), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38488), 56, 0, UchanHR3);
	/* Moving S141_Infos, size 13 from HyperRam at 460776 to (size 13) L2 at 114752 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460776), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 38544), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_e1b513dd using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S128_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S128_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S128_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S128_Conv2d_192x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+9152)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+21120)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+21888)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+22080)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+22464)) /* Infos */
	);
	/* Moving S132_Infos, size 13 from HyperRam at 460728 to (size 13) L2 at 77360 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460728), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1152), 13, 0, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_d6518a03 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_97ec88ff using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S131_Mul_scale using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S131_Mul_shift using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S131_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S131_Conv2d_32x192x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+14976)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+22272)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+22400)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+22432)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+22480)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_5e6564a2, size 6144 from HyperRam at 362880 to (size 6144) L2 at 91184 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 362880), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 14976), 6144, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_413d1607, size 768 from HyperRam at 432576 to (size 768) L2 at 84272 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 432576), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 8064), 768, 0, UchanHR6);
	/* Moving S135_Mul_scale, size 192 from HyperRam at 452512 to (size 192) L2 at 85808 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452512), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9600), 192, 0, UchanHR7);
	/* Moving S135_Mul_shift, size 192 from HyperRam at 452704 to (size 192) L2 at 86000 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452704), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9792), 192, 0, UchanHR8);
	/* Moving S135_Infos, size 13 from HyperRam at 460744 to (size 13) L2 at 86576 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460744), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10368), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_ce350816, size 768 from HyperRam at 433344 to (size 768) L2 at 85040 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 433344), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 8832), 768, 0, UchanHR10);
	/* Moving S138_Mul_scale, size 192 from HyperRam at 452896 to (size 192) L2 at 86192 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 452896), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 9984), 192, 0, UchanHR11);
	/* Moving S138_Mul_shift, size 192 from HyperRam at 453088 to (size 192) L2 at 86384 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 453088), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10176), 192, 0, UchanHR12);
	/* Moving S138_Infos, size 13 from HyperRam at 460760 to (size 13) L2 at 86592 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460760), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 10384), 13, 0, UchanHR13);
	/* Waiting completion of transfer of S132_Infos using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	S132_MatAdd_32x6x6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+13824)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1152)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_5e6564a2 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_413d1607 using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S135_Mul_scale using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S135_Mul_shift using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	/* Waiting completion of transfer of S135_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S135_Conv2d_192x32x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+14976)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+8064)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+9600)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+9792)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10368)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_ce350816 using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S138_Mul_scale using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S138_Mul_shift using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S138_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S138_Conv2d_192x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+10880)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+8832)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+9984)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10176)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+10384)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_666c577b, size 18816 from HyperRam at 221952 to (size 18816) L2 at 84848 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 221952), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 8640), 18816, 0, UchanHR0);
	/* Moving S144_Mul_scale, size 336 from HyperRam at 445120 to (size 336) L2 at 79736 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445120), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3528), 336, 0, UchanHR5);
	/* Moving S144_Mul_shift, size 336 from HyperRam at 445456 to (size 336) L2 at 80072 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445456), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3864), 336, 0, UchanHR6);
	/* Moving S144_Infos, size 13 from HyperRam at 460792 to (size 13) L2 at 80408 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460792), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 4200), 13, 0, UchanHR7);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_9b37a226 using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S141_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S141_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S141_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S141_Conv2d_56x192x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6912)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27456)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory+75896)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+38432)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38488)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+38544)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_3822911a, size 3024 from HyperRam at 408064 to (size 3024) L2 at 103664 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 408064), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 27456), 3024, 0, UchanHR1);
	/* Moving S147_Mul_scale, size 336 from HyperRam at 445792 to (size 336) L2 at 106688 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 445792), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 30480), 336, 0, UchanHR2);
	/* Moving S147_Mul_shift, size 336 from HyperRam at 446128 to (size 336) L2 at 107024 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446128), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 30816), 336, 0, UchanHR3);
	/* Moving S147_Infos, size 13 from HyperRam at 460808 to (size 13) L2 at 107360 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460808), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 31152), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_666c577b using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S144_Mul_scale using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S144_Mul_shift using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S144_Infos using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	S144_Conv2d_336x56x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+8640)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory+20288)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+504)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+3528)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3864)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+4200)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_96b6fec7, size 18816 from HyperRam at 240768 to (size 18816) L2 at 82760 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 240768), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6552), 18816, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_b4c31155, size 224 from HyperRam at 449376 to (size 224) L2 at 101576 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449376), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25368), 224, 0, UchanHR5);
	/* Moving S150_Mul_scale, size 56 from HyperRam at 458480 to (size 56) L2 at 101800 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458480), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25592), 56, 0, UchanHR6);
	/* Moving S150_Mul_shift, size 56 from HyperRam at 458536 to (size 56) L2 at 101856 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458536), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25648), 56, 0, UchanHR7);
	/* Moving S150_Infos, size 13 from HyperRam at 460824 to (size 13) L2 at 101912 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460824), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25704), 13, 0, UchanHR8);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_3822911a using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S147_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S147_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S147_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S147_Conv2d_336x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+504)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+27456)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+21632)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3528)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+30480)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+30816)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+31152)) /* Infos */
	);
	/* Moving S151_Infos, size 13 from HyperRam at 460840 to (size 13) L2 at 77216 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460840), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1008), 13, 0, UchanHR1);
	/* Moving S154_Mul_scale, size 336 from HyperRam at 446464 to (size 336) L2 at 79232 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446464), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3024), 336, 0, UchanHR2);
	/* Moving S157_Mul_scale, size 336 from HyperRam at 447136 to (size 336) L2 at 104600 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447136), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 28392), 336, 0, UchanHR3);
	/* Moving S157_Mul_shift, size 336 from HyperRam at 447472 to (size 336) L2 at 104936 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447472), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 28728), 336, 0, UchanHR4);
	/* Moving S157_Infos, size 13 from HyperRam at 460872 to (size 13) L2 at 105608 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460872), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29400), 13, 0, UchanHR9);
	/* Moving Sequentialmobilenetv2_035_96bl_f3fc8846, size 224 from HyperRam at 449600 to (size 224) L2 at 105272 using event 10 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 449600), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29064), 224, 0, UchanHR10);
	/* Moving S160_Mul_scale, size 56 from HyperRam at 458592 to (size 56) L2 at 105496 using event 11 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458592), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29288), 56, 0, UchanHR11);
	/* Moving S160_Mul_shift, size 56 from HyperRam at 458648 to (size 56) L2 at 105552 using event 12 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 458648), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29344), 56, 0, UchanHR12);
	/* Moving S160_Infos, size 13 from HyperRam at 460888 to (size 13) L2 at 105624 using event 13 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460888), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29416), 13, 0, UchanHR13);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_96b6fec7 using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_b4c31155 using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S150_Mul_scale using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S150_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S150_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S150_Conv2d_56x336x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3528)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6552)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+25368)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+504)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+25592)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+25648)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+25704)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_5073c59c, size 18816 from HyperRam at 259584 to (size 18816) L2 at 82760 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 259584), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6552), 18816, 0, UchanHR0);
	/* Moving S154_Mul_shift, size 336 from HyperRam at 446800 to (size 336) L2 at 79568 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 446800), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3360), 336, 0, UchanHR5);
	/* Moving S154_Infos, size 13 from HyperRam at 460856 to (size 13) L2 at 79904 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460856), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3696), 13, 0, UchanHR6);
	/* Moving Sequentialmobilenetv2_035_96bl_f8cfb438, size 3024 from HyperRam at 411088 to (size 3024) L2 at 101576 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 411088), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25368), 3024, 0, UchanHR7);
	/* Waiting completion of transfer of S151_Infos using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	S151_MatAdd_56x3x3(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+504)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1008)) /* Infos */
	);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_5073c59c using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S154_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S154_Mul_shift using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S154_Infos using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	S154_Conv2d_336x56x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6552)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory+22976)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3360)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3696)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_11f0296b, size 18816 from HyperRam at 278400 to (size 18816) L2 at 82760 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 278400), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6552), 18816, 0, UchanHR0);
	/* Moving Sequentialmobilenetv2_035_96bl_52fbff0b, size 37632 from HyperRam at 184320 to (size 37632) L2 at 232368 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 184320), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 156160), 37632, 0, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_f8cfb438 using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S157_Mul_scale using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S157_Mul_shift using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S157_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S157_Conv2d_336x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+25368)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+24320)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+28392)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+28728)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29400)) /* Infos */
	);
	/* Moving S161_Infos, size 13 from HyperRam at 460904 to (size 13) L2 at 76712 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460904), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 504), 13, 0, UchanHR2);
	/* Moving Sequentialmobilenetv2_035_96bl_6f90883e, size 3024 from HyperRam at 414112 to (size 3024) L2 at 101576 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 414112), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 25368), 3024, 0, UchanHR3);
	/* Moving S167_Mul_scale, size 336 from HyperRam at 448480 to (size 336) L2 at 104600 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448480), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 28392), 336, 0, UchanHR4);
	/* Moving S167_Mul_shift, size 336 from HyperRam at 448816 to (size 336) L2 at 104936 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448816), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 28728), 336, 0, UchanHR5);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_11f0296b using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_f3fc8846 using event 10 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR10);
	/* Waiting completion of transfer of S160_Mul_scale using event 11 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR11);
	/* Waiting completion of transfer of S160_Mul_shift using event 12 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR12);
	/* Waiting completion of transfer of S160_Infos using event 13 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR13);
	S160_Conv2d_56x336x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6552)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+29064)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+29288)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29344)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29416)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_6b6bef1a, size 18816 from HyperRam at 297216 to (size 18816) L2 at 82760 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 297216), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6552), 18816, 0, UchanHR0);
	/* Moving S164_Mul_scale, size 336 from HyperRam at 447808 to (size 336) L2 at 79736 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 447808), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3528), 336, 0, UchanHR6);
	/* Moving S164_Mul_shift, size 336 from HyperRam at 448144 to (size 336) L2 at 80072 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 448144), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 3864), 336, 0, UchanHR7);
	/* Moving S164_Infos, size 13 from HyperRam at 460920 to (size 13) L2 at 80408 using event 8 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460920), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 4200), 13, 0, UchanHR8);
	/* Moving S167_Infos, size 13 from HyperRam at 460936 to (size 13) L2 at 105272 using event 9 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460936), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 29064), 13, 0, UchanHR9);
	/* Waiting completion of transfer of S161_Infos using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	S161_MatAdd_56x3x3(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6048)), /* In1 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* Out */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+504)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96bl_606f6059, size 448 from HyperRam at 438720 to (size 448) L2 at 82256 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 438720), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6048), 448, 0, UchanHR2);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_6b6bef1a using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S164_Mul_scale using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S164_Mul_shift using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	/* Waiting completion of transfer of S164_Infos using event 8 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR8);
	S164_Conv2d_336x56x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6552)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory+25664)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+3528)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3864)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+4200)) /* Infos */
	);
	/* Moving S170_Mul_scale, size 112 from HyperRam at 456224 to (size 112) L2 at 82704 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456224), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6496), 112, 0, UchanHR0);
	/* Moving S170_Mul_shift, size 112 from HyperRam at 456336 to (size 112) L2 at 82816 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456336), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6608), 112, 0, UchanHR6);
	/* Moving S170_Infos, size 13 from HyperRam at 460952 to (size 13) L2 at 82928 using event 7 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460952), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 6720), 13, 0, UchanHR7);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_6f90883e using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S167_Mul_scale using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	/* Waiting completion of transfer of S167_Mul_shift using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S167_Infos using event 9 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR9);
	S167_Conv2d_336x1x3x3_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+25368)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+27008)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+28392)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+28728)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+29064)) /* Infos */
	);
	/* Moving Sequentialmobilenetv2_035_96co_190efb1b, size 143360 from HyperRam at 0 to (size 143360) L2 at 89008 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 0), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 12800), 143360, 0, UchanHR3);
	/* Moving S173_Infos, size 13 from HyperRam at 460968 to (size 13) L2 at 77216 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460968), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 1008), 13, 0, UchanHR4);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_52fbff0b using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96bl_606f6059 using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S170_Mul_scale using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S170_Mul_shift using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	/* Waiting completion of transfer of S170_Infos using event 7 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR7);
	S170_Conv2d_112x336x1x1(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+3024)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+156160)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+6048)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+6496)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6608)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+6720)) /* Infos */
	);
	/* Moving S173_Mul_scale, size 1280 from HyperRam at 421024 to (size 1280) L2 at 232368 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 421024), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 156160), 1280, 0, UchanHR0);
	/* Moving S173_Mul_shift, size 1280 from HyperRam at 422304 to (size 1280) L2 at 233648 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 422304), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 157440), 1280, 0, UchanHR1);
	/* Moving S176_Mul_scale, size 1280 from HyperRam at 423584 to (size 1280) L2 at 234928 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 423584), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 158720), 1280, 0, UchanHR2);
	/* Moving S176_Mul_shift, size 1280 from HyperRam at 424864 to (size 1280) L2 at 236208 using event 5 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 424864), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 160000), 1280, 0, UchanHR5);
	/* Moving S176_Infos, size 13 from HyperRam at 460984 to (size 13) L2 at 237488 using event 6 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 460984), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 161280), 13, 0, UchanHR6);
	/* Waiting completion of transfer of Sequentialmobilenetv2_035_96co_190efb1b using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S173_Mul_scale using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S173_Mul_shift using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S173_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S173_Conv2d_1280x112x1x1_Relu6(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In2 */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+12800)), /* In1 */
		((signed int * __restrict__) (classification_L2_Memory+70776)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1280)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+156160)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+157440)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1008)) /* Infos */
	);
	/* Moving Separable_conv2d_1bias, size 128 from HyperRam at 456096 to (size 128) L2 at 118480 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 456096), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 42272), 128, 0, UchanHR0);
	/* Moving S179_Mul_scale, size 32 from HyperRam at 459568 to (size 32) L2 at 118608 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459568), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 42400), 32, 0, UchanHR1);
	/* Moving S179_Mul_shift, size 32 from HyperRam at 459600 to (size 32) L2 at 118640 using event 3 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 459600), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 42432), 32, 0, UchanHR3);
	/* Moving S179_Infos, size 13 from HyperRam at 461000 to (size 13) L2 at 118672 using event 4 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461000), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 42464), 13, 0, UchanHR4);
	/* Waiting completion of transfer of S176_Mul_scale using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	/* Waiting completion of transfer of S176_Mul_shift using event 5 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR5);
	/* Waiting completion of transfer of S176_Infos using event 6 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR6);
	S176_Conv2d_1280x1x3x3(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1280)), /* In */
		((signed char * __restrict__) (classification_L3_Memory+316032)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory+0)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+158720)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+160000)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+161280)) /* Infos */
	);
	/* Waiting completion of transfer of Separable_conv2d_1bias using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S179_Mul_scale using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S179_Mul_shift using event 3 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR3);
	/* Waiting completion of transfer of S179_Infos using event 4 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR4);
	S179_Conv2d_32x1280x1x1_Relu(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L2_Memory+29816)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+42272)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1280)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+42400)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+42432)), /* ScaleN */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+42464)) /* Infos */
	);
	/* Moving Densebias, size 8 from HyperRam at 461124 to (size 8) L2 at 76324 using event 0 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461124), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 116), 8, 0, UchanHR0);
	/* Moving S183_Mul_scale, size 2 from HyperRam at 461160 to (size 2) L2 at 76332 using event 1 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461160), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 124), 2, 0, UchanHR1);
	/* Moving S183_Mul_shift, size 2 from HyperRam at 461164 to (size 2) L2 at 76336 using event 2 */
	AT_HYPERRAM_CL_COPY(&HyperRam, ((AT_HYPERRAM_EXT_ADDR_TYPE) classification_L3_Memory + 461164), ((AT_HYPERRAM_INT_ADDR_TYPE) classification_L2_Memory_Dyn + 128), 2, 0, UchanHR2);
	S180_Op_MEAN_0_72(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+1280)), /* In */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* Out */
		((signed char * __restrict__) (classification_L3_Memory+461016)) /* Infos */
	);
	/* Waiting completion of transfer of Densebias using event 0 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR0);
	/* Waiting completion of transfer of S183_Mul_scale using event 1 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR1);
	/* Waiting completion of transfer of S183_Mul_shift using event 2 */
	AT_HYPERRAM_CL_WAIT(&HyperRam, UchanHR2);
	S183_Linear_2x32(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+0)), /* In */
		((signed char * __restrict__) (classification_L3_Memory+458304)), /* Filter */
		((signed int * __restrict__) (classification_L2_Memory_Dyn+116)), /* Bias */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32)), /* Out */
		((unsigned char * __restrict__) (classification_L2_Memory_Dyn+124)), /* Scale */
		((signed char * __restrict__) (classification_L2_Memory_Dyn+128)), /* ScaleN */
		((signed char * __restrict__) (classification_L3_Memory+461032)) /* Infos */
	);
	S184_SoftMax(
		((signed char * __restrict__) (classification_L2_Memory_Dyn+32)), /* In */
		((signed short * __restrict__) Output_1), /* Out */
		((signed char * __restrict__) (classification_L3_Memory+461048)) /* Infos */
	);
	return 0;
}
