/**
  ******************************************************************************
  * @file    eis_network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2024-11-22T10:09:41+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "eis_network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_eis_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_eis_network_weights_array_u64[1] = {
  0xbe21f9a83f8d4f33U,
};


ai_handle g_eis_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_eis_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

