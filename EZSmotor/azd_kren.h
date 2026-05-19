#ifndef AZD_KREN_H
#define AZD_KREN_H

/* ============================================================
 *  azd_kren.h
 *  AZD-KREN Ethernet Driver — public API
 * ============================================================ */

#include <stdint.h>
#include <modbus.h>

/* ── Driver handle ──────────────────────────────────────────────────────── */
typedef struct {
    modbus_t *ctx;
    int       slave_id;
} AZDKREN;

/* ── Status snapshot ────────────────────────────────────────────────────── */
typedef struct {
    int      ready;        /* 1 = driver ready to accept command  */
    int      in_pos;       /* 1 = target position reached         */
    int      dcmd_rdy;     /* 1 = direct command ready            */
    int      alarm;        /* 1 = alarm active                    */
    uint16_t alarm_code;   /* alarm code (0 = no alarm)           */
} AZDStatus;

/* ── Connection ─────────────────────────────────────────────────────────── */
int   azdkren_connect   (AZDKREN *drv, const char *ip, int port, int slave_id);
void  azdkren_disconnect(AZDKREN *drv);

/* ── Status / monitoring ────────────────────────────────────────────────── */
int   azdkren_get_status  (AZDKREN *drv, AZDStatus *out);
float azdkren_get_position(AZDKREN *drv);   /* mm,   -1.0 on error */
float azdkren_get_speed   (AZDKREN *drv);   /* mm/s, -1.0 on error */
void  azdkren_print_status(AZDKREN *drv);

/* ── Wait helpers ───────────────────────────────────────────────────────── */
int   azdkren_wait_ready      (AZDKREN *drv, float timeout_s);
int   azdkren_wait_in_position(AZDKREN *drv, float timeout_s);

/* ── Control ────────────────────────────────────────────────────────────── */
int   azdkren_reset_alarm  (AZDKREN *drv);
int   azdkren_stop         (AZDKREN *drv);
int   azdkren_motor_free   (AZDKREN *drv);
int   azdkren_motor_on     (AZDKREN *drv);
int   azdkren_home         (AZDKREN *drv, int wait);

/*
 * azdkren_set_home_here — declare current position as the coordinate origin.
 *   The motor does NOT move; only the coordinate system is redefined.
 *   save_to_nvm: 1 → persist to flash (survives power-off), 0 → RAM only.
 */
int   azdkren_set_home_here(AZDKREN *drv, int save_to_nvm);

/* ── Motion (Direct Data Operation / TRIG method) ───────────────────────── */
/*
 * azdkren_move_abs — absolute position move
 *   pos_mm       : target position from home [mm]
 *   speed_mm_s   : travel speed [mm/s]
 *   acc_hz_s     : acceleration rate [Hz/s]  (steps/s²)
 *   dec_hz_s     : deceleration rate [Hz/s]
 *   current_pct  : motor operating current [0–100 %]
 *   wait         : 1 = block until IN-POS, 0 = return immediately
 *   returns 0 on success, -1 on error / timeout / alarm
 */
int   azdkren_move_abs(AZDKREN *drv,
                       float pos_mm,   float speed_mm_s,
                       float acc_hz_s, float dec_hz_s,
                       float current_pct, int wait);

/*
 * azdkren_move_rel — relative move from current feedback position
 *   delta_mm : displacement [mm]  (positive = forward, negative = reverse)
 */
int   azdkren_move_rel(AZDKREN *drv,
                       float delta_mm, float speed_mm_s,
                       float acc_hz_s, float dec_hz_s,
                       float current_pct, int wait);

#endif /* AZD_KREN_H */
