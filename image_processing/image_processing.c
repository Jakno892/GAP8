/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * AI-deck examples
 *
 * Copyright (C) 2021 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * @file image_processing.c
 *
 *  
 */

#include "image_processing.h"
#include "pmsis.h"
#include "bsp/bsp.h"
#include "cpx.h"

#define CAM_WIDTH 324
#define CAM_HEIGHT 244

#define CHANNELS 1
#define IO RGB888_IO
#define CAT_LEN sizeof(uint32_t)

void control(void)
{
    pi_bsp_init();
    cpxInit();

    cpxEnableFunction(CPX_F_APP);
    cpxPrintToConsole(LOG_TO_CRTP, "Starting counter bouncer\n");

    uint8_t counter = 100; // Height above ground in cm
    bool in_position;

    cpxInitRoute(CPX_T_GAP8, CPX_T_STM32, CPX_F_APP, &txPacket.route);
    txPacket.data[0] = counter;
    txPacket.dataLength = 1;

    cpxSendPacketBlocking(&txPacket);

    while (counter)
    {
        // cpxPrintToConsole(LOG_TO_CRTP, counter);
        cpxReceivePacketBlocking(CPX_F_APP, &rxPacket);
        in_position = rxPacket.data[0];
        if(in_position)
        {
            // Send the height value to the STM
            cpxInitRoute(CPX_T_GAP8, CPX_T_STM32, CPX_F_APP, &txPacket.route);
            txPacket.data[0] = counter;
            txPacket.dataLength = 1;

            cpxSendPacketBlocking(&txPacket);
            counter -= 5;
        }

        pi_time_wait_us(5*1000*1000);
    }

}

int main(void)
{
  return pmsis_kickoff((void *)control);
}