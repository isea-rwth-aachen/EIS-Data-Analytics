/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
__attribute__((weak)) void _close(void) {
}
__attribute__((weak)) void _lseek(void) {
}
__attribute__((weak)) void _read(void) {
}
__attribute__((weak)) void _write(void) {
}
__attribute__((weak)) void _fstat(void) {
}
__attribute__((weak)) void _getpid(void) {
}
__attribute__((weak)) void _isatty(void) {
}
__attribute__((weak)) void _kill(void) {
}

#include "stdbool.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "arm_math.h"

#include "test_arrays.h"
#include "ai_datatypes_defines.h"
#include "eis_network.h"
#include "eis_network_data_params.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
ai_handle eis_network;

float model_in_data[AI_EIS_NETWORK_IN_1_SIZE];
float model_out_data[AI_EIS_NETWORK_OUT_1_SIZE];

ai_u8 activations[AI_EIS_NETWORK_DATA_ACTIVATIONS_SIZE];

ai_buffer *ai_input;
ai_buffer *ai_output;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
static void AI_Init(void);
static void AI_Run(float *pIn, float *pOut);

static void arm_arrhenius_correction(float *data, int block_size);
static void arm_arrhenius_correction_with_factor(float *data, int block_size,
		float arrhenius_b, float arrhenius_c);

static void arm_compute_power(float *data, float *result, int max_exponent,
		int block_size);

static void arm_min_max_scaler(float *data, int block_size, float min,
		float max);
static void arm_inverse_min_max_scaler(float *data, int block_size, float min,
		float max);

static void arm_standard_scaler(float *data, int block_size, float mean,
		float std);
static void arm_inverse_standard_scaler(float *data, int block_size, float mean,
		float std);

static float compute_max(float *array, int length, int width);
static float compute_mean(float *array, int length, int width);
static float compute_std(float *array, int length, int width);
static float compute_rms(float *array, int length, int width);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
	// UART Buffer
	char buf[256];
	int buf_len = 0;
	// Execution Time stamps
	uint16_t timestamp_before_input, timestamp_model_start,
			timestamp_model_finished, timestamp_output;

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM14_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
	// Start Timer/counter
	HAL_TIM_Base_Start(&htim14);

	// Start the Program
	buf_len = sprintf(buf, "\r\n\r\nEIS AI Evaluation\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);

	AI_Init();

	// Get the Size of the Input and Output Data
	int test_input_length = (int) (sizeof(test_input) / sizeof(test_input[0]));
	int test_input_width = (int) (sizeof(test_input[0])
			/ sizeof(test_input[0][0]));

	int test_output_length =
			(int) (sizeof(test_output) / sizeof(test_output[0]));
	int test_output_width = (int) (sizeof(test_output[0])
			/ sizeof(test_output[0][0]));

	// Create Buffer for Input (will be modified), Results and Differences
	float test_input_tmp[test_input_length][test_input_width];
	float test_input_tmp_poly[test_input_length][AI_EIS_NETWORK_IN_1_SIZE];
	float test_output_predicted[test_output_length][test_output_width];
	float diff[test_output_length][test_output_width];

	HAL_GPIO_WritePin(GPIOB, LD1_Pin | LD3_Pin | LD2_Pin, GPIO_PIN_SET);
	HAL_Delay(1000);
	// Startup finished
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1) {
		// Prepare/Reset data
		memcpy(test_input_tmp, test_input, sizeof(test_input));
		HAL_GPIO_WritePin(GPIOB, LD1_Pin | LD3_Pin | LD2_Pin, GPIO_PIN_RESET);

		// Evaluate all Input values individually
		for (int i = 0; i < test_input_length; i++) {
			timestamp_before_input = (uint16_t) htim14.Instance->CNT;
			// First Scale the Data
			if (use_arrhenius_correction) {
				arm_arrhenius_correction(&test_input_tmp[i][0],
						test_input_width);
			} else if (use_arrhenius_correction_with_factor) {
				arm_arrhenius_correction_with_factor(&test_input_tmp[i][0],
						test_input_width, arrhenius_b, arrhenius_c);
			}

			arm_compute_power(&test_input_tmp[i][0], &test_input_tmp_poly[i][0],
					polynomial_degree, test_input_width);

			if (use_min_max_scaler) {
				arm_min_max_scaler(&test_input_tmp_poly[i][0],
				AI_EIS_NETWORK_IN_1_SIZE, min_max_scaler_x_min,
						min_max_scaler_x_max);
			} else if (use_standard_scaler) {
				arm_standard_scaler(&test_input_tmp_poly[i][0],
				AI_EIS_NETWORK_IN_1_SIZE, standard_scaler_x_mean,
						standard_scaler_x_std);
			}

			memcpy(model_in_data, &test_input_tmp_poly[i],
					sizeof(test_input[0][0]) * AI_EIS_NETWORK_IN_1_SIZE);

			// Run the model
			timestamp_model_start = (uint16_t) htim14.Instance->CNT;
			AI_Run(model_in_data, model_out_data);
			timestamp_model_finished = (uint16_t) htim14.Instance->CNT;

			memcpy(&test_output_predicted[i], model_out_data,
					sizeof(test_output[0]));

			// Scale output, if it is scaled
			if (scale_y_data) {
				if (use_min_max_scaler) {
					arm_inverse_min_max_scaler(&test_output_predicted[i][0],
					AI_EIS_NETWORK_OUT_1_SIZE, min_max_scaler_y_min,
							min_max_scaler_y_max);
				} else if (use_standard_scaler) {
					arm_inverse_standard_scaler(&test_output_predicted[i][0],
					AI_EIS_NETWORK_OUT_1_SIZE, standard_scaler_y_mean,
							standard_scaler_y_std);
				}
			}

			timestamp_output = (uint16_t) htim14.Instance->CNT;

			// Send Results
			arm_sub_f32(&test_output_predicted[i][0], &test_output[i][0],
					&diff[i][0], test_output_width);
			buf_len = sprintf(buf, "Real Value: ");
			HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			for (int output_width_index = 0;
					output_width_index < test_output_width;
					output_width_index++) {
				buf_len = sprintf(buf, "%f ",
						test_output[i][output_width_index]);
				HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			}
			buf_len = sprintf(buf, "\t| Estimated Value: ");
			HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			for (int output_width_index = 0;
					output_width_index < test_output_width;
					output_width_index++) {
				buf_len = sprintf(buf, "%f ",
						test_output_predicted[i][output_width_index]);
				HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			}
			buf_len = sprintf(buf, "\t| Diff: ");
			HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			for (int output_width_index = 0;
					output_width_index < test_output_width;
					output_width_index++) {
				buf_len = sprintf(buf, "%f ", diff[i][output_width_index]);
				HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			}
			buf_len =
					sprintf(buf,
							"\t| Total Duration: %u us\t| Model Duration %u us\r\n",
							(uint16_t) (timestamp_output
									- timestamp_before_input),
							(uint16_t) (timestamp_model_finished
									- timestamp_model_start));
			HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
			HAL_GPIO_TogglePin(GPIOB, LD1_Pin);
		}

		// Evaluate the Error and start again after a short delay
		float mean_value = compute_mean(&diff[0][0], test_output_length,
				test_output_width);
		float std_value = compute_std(&diff[0][0], test_output_length,
				test_output_width);
		float rms_value = compute_rms(&diff[0][0], test_output_length,
				test_output_width);
		float max_value = compute_max(&diff[0][0], test_output_length,
				test_output_width);
		buf_len =
				sprintf(buf,
						"Maximum Difference:\t %f \r\nMean Difference:\t %f \r\nStd. Difference:\t %f \r\nRMS  Difference:\t %f \r\n",
						max_value, mean_value, std_value, rms_value);
		HAL_UART_Transmit(&huart3, (uint8_t*) buf, buf_len, 100);
		HAL_GPIO_WritePin(GPIOB, LD1_Pin | LD2_Pin, GPIO_PIN_SET);
		HAL_Delay(4000);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
static void AI_Init(void) {
	ai_error err;
	const ai_handle act_addr[] = { activations };
	err = ai_eis_network_create_and_init(&eis_network, act_addr, NULL);
	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
	ai_input = ai_eis_network_inputs_get(eis_network, NULL);
	ai_output = ai_eis_network_outputs_get(eis_network, NULL);
}

static void AI_Run(float *pIn, float *pOut) {
	ai_i32 batch;
	ai_error err;
	ai_input[0].data = AI_HANDLE_PTR(pIn);
	ai_output[0].data = AI_HANDLE_PTR(pOut);
	batch = ai_eis_network_run(eis_network, ai_input, ai_output);
	if (batch != 1) {
		err = ai_eis_network_get_error(eis_network);
		printf("AI ai_network_run error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
}

static void arm_arrhenius_correction(float *data, int block_size) {
	arm_vlog_f32(data, data, block_size);
	arm_negate_f32(data, data, block_size);
}

static void arm_arrhenius_correction_with_factor(float *data, int block_size,
		float arrhenius_b, float arrhenius_c) {
	arm_scale_f32(data, arrhenius_c, data, block_size);
	arm_vlog_f32(data, data, block_size);
	arm_scale_f32(data, arrhenius_b, data, block_size);
}

static void arm_compute_power(float *data, float *result, int max_exponent,
		int block_size) {
	memcpy(&result[0], data, block_size * sizeof(result[0]));
	for (int exponent = 1; exponent < max_exponent; exponent++) {
		arm_vlog_f32(data, &result[exponent], block_size);
		arm_scale_f32(&result[exponent], exponent + 1.0, &result[exponent],
				block_size);
		arm_vexp_f32(&result[exponent], &result[exponent], block_size);

	}
}

static void arm_min_max_scaler(float *data, int block_size, float min,
		float max) {
	arm_offset_f32(data, -min, data, block_size);
	arm_scale_f32(data, 1.0 / (max - min), data, block_size);
}

static void arm_inverse_min_max_scaler(float *data, int block_size, float min,
		float max) {
	arm_scale_f32(data, (max - min), data, block_size);
	arm_offset_f32(data, min, data, block_size);

}

static void arm_standard_scaler(float *data, int block_size, float mean,
		float std) {
	arm_offset_f32(data, -mean, data, block_size);
	arm_scale_f32(data, 1.0 / (std), data, block_size);
}

static void arm_inverse_standard_scaler(float *data, int block_size, float mean,
		float std) {
	arm_scale_f32(data, std, data, block_size);
	arm_offset_f32(data, mean, data, block_size);
}

static float compute_max(float *array, int length, int width) {
	float max = 0.0;
	uint32_t max_idx = 0;
	arm_abs_f32(array, array, length * width);
	arm_max_f32(array, length * width, &max, &max_idx);
	return max;
}

static float compute_mean(float *array, int length, int width) {
	float mean = 0.0;
	arm_mean_f32(array, length * width, &mean);
	return mean;
}

static float compute_std(float *array, int length, int width) {
	float std = 0.0;
	arm_std_f32(array, length * width, &std);
	return std;
}

static float compute_rms(float *array, int length, int width) {
	float rms = 0.0;
	arm_rms_f32(array, length * width, &rms);
	return rms;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
