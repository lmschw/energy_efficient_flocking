/**
 * Crazyflie app for the Hebbian-controller deployment on CrazyThymio robots with Lighthouse
 * positioning. Derived from examples/app_share_pos/src/share_pos.c of
 * https://github.com/tugayalperen/CrazyThymio-firmware (GPL-3.0, Bitcraze AB / the CrazyThymio
 * authors), reduced to what the Hebbian controller needs: instead of computing the 4-quadrant
 * distance/heading itself, this app hands the Raspberry Pi
 *   - the robot's own CENTER position (board position + mounting offset, see board_offset_x/y, params hebb.offx/offy),
 *   - the latest position of every other robot heard over the radio (P2P broadcast),
 * and the Pi computes range/bearing sensing exactly as the simulation does
 * (ants26_replication/hardware_deployment/sensor_model.py).
 *
 * Robot id = lowest byte of its Crazyflie radio address: any value 1..255, unique per robot.
 * The table holds up to N_PEERS (10) robots in slots assigned in order of first contact; each
 * slot logs the id of the robot it currently holds (0 = empty), so the Pi maps slots to robots
 * by id, never by slot position. Positions are int16 millimetres; -32768 and id 0 mean the slot
 * is empty (nothing heard from it in the last NEIGHBOR_TIMEOUT_MS).
 *
 * Log groups:  ctr (own center x, y [m, float]),
 *              nbA (x1,y1,id1 .. x5,y5,id5), nbB (x6,y6,id6 .. x10,y10,id10)
 *              x/y: int16 mm, id: uint8
 */
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "radiolink.h"
#include "configblock.h"
#include "log.h"
#include "param.h"

#define DEBUG_MODULE "HEBBIAN"
#include "debug.h"

#define N_PEERS 10
#define NO_NEIGHBOR (-32768)
#define NEIGHBOR_TIMEOUT_MS 1000
#define LOOP_MS 50

// Position of the Crazyflie board relative to the Thymio's center, in the board frame
// (x forward, y left), in metres. Values of the upstream CrazyThymio rig -- MEASURE on yours.
// Vector from the Crazyflie board to the Thymio's centre of rotation, in the board frame (x forward, y left), metres.
// Runtime PARAMETERS (group "hebb": offx, offy) -- each Pi sets its own robot's measured value at start-up
// (crazythymio/lighthouse_deployment/controller_config.py BOARD_OFFSET_M, measured with tools/measure_offset.sh).
// The default is the fleet median measured 2026-10-04. The upstream rig's (-0.09, +0.04) was ~9 cm wrong on every robot.
static float board_offset_x = -0.0025f;
static float board_offset_y = 0.021f;

typedef struct {
  uint8_t id;
  float x;
  float y;
  float h;
  uint8_t l;
} _coords;   // same layout as upstream so both firmwares can coexist on one radio channel

static int16_t nb_x[N_PEERS];
static int16_t nb_y[N_PEERS];
static uint8_t nb_id[N_PEERS];     // 0 = slot empty
static TickType_t nb_tick[N_PEERS];
static bool nb_heard[N_PEERS];
static float ctr_x = 0.0f;
static float ctr_y = 0.0f;

static void p2pcallbackHandler(P2PPacket *p)
{
  _coords other;
  memcpy(&other, p->data, sizeof(other));
  if (other.id == 0) {
    return;
  }
  // Slot already holding this id, else the first free slot, else drop (more than N_PEERS robots).
  int i = -1;
  for (int k = 0; k < N_PEERS; k++) {
    if (nb_heard[k] && nb_id[k] == other.id) {
      i = k;
      break;
    }
  }
  if (i < 0) {
    for (int k = 0; k < N_PEERS; k++) {
      if (!nb_heard[k]) {
        i = k;
        break;
      }
    }
  }
  if (i < 0) {
    return;
  }
  nb_id[i] = other.id;
  // Transmitted coordinates are already the sender's center position (see appMain()).
  nb_x[i] = (int16_t)(other.x * 1000.0f);
  nb_y[i] = (int16_t)(other.y * 1000.0f);
  nb_tick[i] = xTaskGetTickCount();
  nb_heard[i] = true;
}

void appMain()
{
  for (int i = 0; i < N_PEERS; i++) {
    nb_x[i] = NO_NEIGHBOR;
    nb_y[i] = NO_NEIGHBOR;
    nb_id[i] = 0;
    nb_heard[i] = false;
  }

  logVarId_t idX = logGetVarId("stateEstimate", "x");
  logVarId_t idY = logGetVarId("stateEstimate", "y");
  logVarId_t idYaw = logGetVarId("stateEstimate", "yaw");

  uint8_t my_id = (uint8_t)(configblockGetRadioAddress() & 0xff);
  DEBUG_PRINT("Hebbian share_pos app, radio id %d\n", my_id);

  PARAM_GROUP_START(hebb)
  PARAM_ADD(PARAM_FLOAT, offx, &board_offset_x)
  PARAM_ADD(PARAM_FLOAT, offy, &board_offset_y)
  PARAM_GROUP_STOP(hebb)

  LOG_GROUP_START(ctr)
  LOG_ADD(LOG_FLOAT, x, &ctr_x)
  LOG_ADD(LOG_FLOAT, y, &ctr_y)
  LOG_GROUP_STOP(ctr)

  LOG_GROUP_START(nbA)
  LOG_ADD(LOG_INT16, x1, &nb_x[0])
  LOG_ADD(LOG_INT16, y1, &nb_y[0])
  LOG_ADD(LOG_UINT8, id1, &nb_id[0])
  LOG_ADD(LOG_INT16, x2, &nb_x[1])
  LOG_ADD(LOG_INT16, y2, &nb_y[1])
  LOG_ADD(LOG_UINT8, id2, &nb_id[1])
  LOG_ADD(LOG_INT16, x3, &nb_x[2])
  LOG_ADD(LOG_INT16, y3, &nb_y[2])
  LOG_ADD(LOG_UINT8, id3, &nb_id[2])
  LOG_ADD(LOG_INT16, x4, &nb_x[3])
  LOG_ADD(LOG_INT16, y4, &nb_y[3])
  LOG_ADD(LOG_UINT8, id4, &nb_id[3])
  LOG_ADD(LOG_INT16, x5, &nb_x[4])
  LOG_ADD(LOG_INT16, y5, &nb_y[4])
  LOG_ADD(LOG_UINT8, id5, &nb_id[4])
  LOG_GROUP_STOP(nbA)

  LOG_GROUP_START(nbB)
  LOG_ADD(LOG_INT16, x6, &nb_x[5])
  LOG_ADD(LOG_INT16, y6, &nb_y[5])
  LOG_ADD(LOG_UINT8, id6, &nb_id[5])
  LOG_ADD(LOG_INT16, x7, &nb_x[6])
  LOG_ADD(LOG_INT16, y7, &nb_y[6])
  LOG_ADD(LOG_UINT8, id7, &nb_id[6])
  LOG_ADD(LOG_INT16, x8, &nb_x[7])
  LOG_ADD(LOG_INT16, y8, &nb_y[7])
  LOG_ADD(LOG_UINT8, id8, &nb_id[7])
  LOG_ADD(LOG_INT16, x9, &nb_x[8])
  LOG_ADD(LOG_INT16, y9, &nb_y[8])
  LOG_ADD(LOG_UINT8, id9, &nb_id[8])
  LOG_ADD(LOG_INT16, x10, &nb_x[9])
  LOG_ADD(LOG_INT16, y10, &nb_y[9])
  LOG_ADD(LOG_UINT8, id10, &nb_id[9])
  LOG_GROUP_STOP(nbB)

  p2pRegisterCB(p2pcallbackHandler);
  vTaskDelay(M2T(1000));

  static P2PPacket reply;
  reply.port = 0x00;
  _coords self;
  memset(&self, 0, sizeof(self));
  self.id = my_id;

  while (1) {
    float yaw = logGetFloat(idYaw) * 0.0174532f;   // degrees -> rad
    float bx = logGetFloat(idX);
    float by = logGetFloat(idY);
    ctr_x = bx + cosf(yaw) * board_offset_x - sinf(yaw) * board_offset_y;
    ctr_y = by + sinf(yaw) * board_offset_x + cosf(yaw) * board_offset_y;

    self.x = ctr_x;
    self.y = ctr_y;
    self.h = yaw;
    memcpy(reply.data, &self, sizeof(self));
    reply.size = sizeof(self) + 1;
    radiolinkSendP2PPacketBroadcast(&reply);

    TickType_t now = xTaskGetTickCount();
    for (int i = 0; i < N_PEERS; i++) {
      if (nb_heard[i] && (now - nb_tick[i]) > M2T(NEIGHBOR_TIMEOUT_MS)) {
        nb_heard[i] = false;
        nb_x[i] = NO_NEIGHBOR;
        nb_y[i] = NO_NEIGHBOR;
        nb_id[i] = 0;
      }
    }
    vTaskDelay(M2T(LOOP_MS));
  }
}
