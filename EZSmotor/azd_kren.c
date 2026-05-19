/* ============================================================
 *  azd_kren.c
 *  AZD-KREN Ethernet Driver — implementation
 * ============================================================ */

#include "azd_kren.h"
#include "registers.h"

#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>   /* usleep */

/* ── Internal helpers ───────────────────────────────────────────────────── */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_ms(int ms)
{
    usleep((useconds_t)(ms * 1000));
}

/* Signed 32-bit ↔ two unsigned 16-bit Modbus words (little-endian word order) */
static void to_u32_words(int32_t value, uint16_t *lo, uint16_t *hi)
{
    uint32_t v = (uint32_t)value;
    *lo = (uint16_t)(v & 0xFFFF);
    *hi = (uint16_t)(v >> 16);
}

static int32_t from_u32_words(uint16_t lo, uint16_t hi)
{
    uint32_t raw = ((uint32_t)hi << 16) | (uint32_t)lo;
    return (int32_t)raw;
}

/* ── Low-level Modbus wrappers ──────────────────────────────────────────── */

static int read_regs(AZDKREN *drv, int addr, int count, uint16_t *dest)
{
    int rc = modbus_read_registers(drv->ctx, addr, count, dest);
    if (rc != count) {
        fprintf(stderr, "[Modbus] Read error @ 0x%04X: %s\n",
                addr, modbus_strerror(errno));
        return -1;
    }
    return 0;
}

static int write_reg(AZDKREN *drv, int addr, uint16_t value)
{
    /* Use FC16 (write multiple) with count=1 — some AZD-KREN firmware
     * rejects FC6 (write single) on control registers. */
    int rc = modbus_write_registers(drv->ctx, addr, 1, &value);
    if (rc != 1) {
        fprintf(stderr, "[Modbus] Write error @ 0x%04X: %s\n",
                addr, modbus_strerror(errno));
        return -1;
    }
    return 0;
}

static int write_regs(AZDKREN *drv, int addr, const uint16_t *values, int count)
{
    int rc = modbus_write_registers(drv->ctx, addr, count, values);
    if (rc != count) {
        fprintf(stderr, "[Modbus] Multi-write error @ 0x%04X: %s\n",
                addr, modbus_strerror(errno));
        return -1;
    }
    return 0;
}

/* Assert a bit for hold_ms, then clear (edge-triggered signal) */
static int toggle_bit(AZDKREN *drv, int reg, uint16_t bit, int hold_ms)
{
    if (write_reg(drv, reg, bit) != 0) return -1;
    sleep_ms(hold_ms);
    return write_reg(drv, reg, 0);
}

/* ── Connection ─────────────────────────────────────────────────────────── */

int azdkren_connect(AZDKREN *drv, const char *ip, int port, int slave_id)
{
    drv->slave_id = slave_id;
    drv->ctx = modbus_new_tcp(ip, port);
    if (drv->ctx == NULL) {
        fprintf(stderr, "Failed to allocate Modbus context: %s\n",
                modbus_strerror(errno));
        return -1;
    }
    modbus_set_slave(drv->ctx, slave_id);
    if (modbus_connect(drv->ctx) == -1) {
        fprintf(stderr, "Connection failed (%s:%d): %s\n",
                ip, port, modbus_strerror(errno));
        modbus_free(drv->ctx);
        drv->ctx = NULL;
        return -1;
    }
    printf("Connected to AZD-KREN @ %s:%d  slave=%d\n", ip, port, slave_id);
    return 0;
}

void azdkren_disconnect(AZDKREN *drv)
{
    if (drv->ctx) {
        modbus_close(drv->ctx);
        modbus_free(drv->ctx);
        drv->ctx = NULL;
    }
    printf("Disconnected from AZD-KREN.\n");
}

/* ── Status / monitoring ────────────────────────────────────────────────── */

int azdkren_get_status(AZDKREN *drv, AZDStatus *out)
{
    uint16_t regs[2];
    if (read_regs(drv, REG_FIXED_IO_OUT, 2, regs) != 0) return -1;
    out->ready      = (regs[0] & BIT_READY)    ? 1 : 0;
    out->in_pos     = (regs[0] & BIT_IN_POS)   ? 1 : 0;
    out->dcmd_rdy   = (regs[0] & BIT_DCMD_RDY) ? 1 : 0;
    out->alarm      = (regs[0] & BIT_ALM_A)    ? 1 : 0;
    out->alarm_code = regs[1];
    return 0;
}

float azdkren_get_position(AZDKREN *drv)
{
    uint16_t regs[2];
    if (read_regs(drv, REG_FDBK_POS_L, 2, regs) != 0) return -1.0f;
    int32_t steps = from_u32_words(regs[0], regs[1]);
    return (float)steps / STEPS_PER_MM;
}

float azdkren_get_speed(AZDKREN *drv)
{
    uint16_t regs[2];
    if (read_regs(drv, REG_FDBK_SPD_L, 2, regs) != 0) return -1.0f;
    int32_t hz = from_u32_words(regs[0], regs[1]);
    return (float)hz / STEPS_PER_MM;
}

void azdkren_print_status(AZDKREN *drv)
{
    AZDStatus st;
    if (azdkren_get_status(drv, &st) != 0) {
        printf("Could not read status.\n");
        return;
    }
    float pos = azdkren_get_position(drv);
    printf("Status → ready=%d  in_pos=%d  dcmd_rdy=%d  alarm=%d",
           st.ready, st.in_pos, st.dcmd_rdy, st.alarm);
    if (st.alarm)
        printf("  (alarm_code=0x%04X)", st.alarm_code);
    if (pos >= 0.0f)
        printf("  position=%.3f mm", pos);
    printf("\n");
}

/* ── Wait helpers ───────────────────────────────────────────────────────── */

int azdkren_wait_ready(AZDKREN *drv, float timeout_s)
{
    double deadline = now_s() + (double)timeout_s;
    AZDStatus st;
    while (now_s() < deadline) {
        if (azdkren_get_status(drv, &st) != 0) { sleep_ms(100); continue; }
        if (st.alarm) {
            fprintf(stderr, "Alarm while waiting READY: code=0x%04X\n",
                    st.alarm_code);
            return -1;
        }
        if (st.ready) return 0;
        sleep_ms(50);
    }
    fprintf(stderr, "Timeout (%.1fs) waiting for READY.\n", timeout_s);
    return -1;
}

int azdkren_wait_in_position(AZDKREN *drv, float timeout_s)
{
    double deadline = now_s() + (double)timeout_s;
    AZDStatus st;
    while (now_s() < deadline) {
        if (azdkren_get_status(drv, &st) != 0) { sleep_ms(100); continue; }
        if (st.alarm) {
            fprintf(stderr, "Alarm during move: code=0x%04X\n", st.alarm_code);
            return -1;
        }
        if (st.in_pos) return 0;
        sleep_ms(50);
    }
    fprintf(stderr, "Timeout (%.1fs) waiting for IN-POS.\n", timeout_s);
    return -1;
}

/* ── Control ────────────────────────────────────────────────────────────── */

int azdkren_reset_alarm(AZDKREN *drv)
{
    if (toggle_bit(drv, REG_FIXED_IO_IN, BIT_ALM_RST, 50) != 0) return -1;
    sleep_ms(100);
    printf("Alarm reset issued.\n");
    return 0;
}

int azdkren_stop(AZDKREN *drv)
{
    if (toggle_bit(drv, REG_FIXED_IO_IN, BIT_STOP, 20) != 0) return -1;
    printf("Stop issued.\n");
    return 0;
}

int azdkren_motor_free(AZDKREN *drv)
{
    if (write_reg(drv, REG_FIXED_IO_IN, BIT_FREE) != 0) return -1;
    printf("Motor FREE (torque off).\n");
    return 0;
}

int azdkren_motor_on(AZDKREN *drv)
{
    if (write_reg(drv, REG_FIXED_IO_IN, 0) != 0) return -1;
    printf("Motor ON (torque on).\n");
    return 0;
}

int azdkren_home(AZDKREN *drv, int wait)
{
    if (azdkren_wait_ready(drv, 10.0f) != 0) return -1;
    if (toggle_bit(drv, REG_FIXED_IO_IN, BIT_HOME, 50) != 0) return -1;
    printf("Home (origin return) issued.\n");
    if (!wait) return 0;
    int rc = azdkren_wait_in_position(drv, 60.0f);
    if (rc == 0)
        printf("Homing complete. Position: %.3f mm\n", azdkren_get_position(drv));
    return rc;
}

int azdkren_set_home_here(AZDKREN *drv, int save_to_nvm)
{
    float before = azdkren_get_position(drv);
    if (write_reg(drv, REG_P_PRESET, 1) != 0) return -1;
    sleep_ms(100);
    write_reg(drv, REG_P_PRESET, 0);
    float after = azdkren_get_position(drv);
    printf("Home set here.  (was %.3f mm → now %.3f mm)\n", before, after);
    if (save_to_nvm) {
        sleep_ms(200);
        write_reg(drv, REG_NVM_WRITE, 1);
        sleep_ms(1000);
        write_reg(drv, REG_NVM_WRITE, 0);
        printf("Saved to NV-memory (persists after power-off).\n");
    }
    return 0;
}

/* ── Motion ─────────────────────────────────────────────────────────────── */

static int direct_move(AZDKREN *drv,
                       float position_mm, float speed_mm_s,
                       float acc_hz_s,    float dec_hz_s,
                       float current_pct, int op_type, int wait)
{
    if (azdkren_wait_ready(drv, 10.0f) != 0) return -1;

    int32_t pos_steps = (int32_t)(position_mm * STEPS_PER_MM);
    int32_t spd_hz    = (int32_t)(speed_mm_s  * STEPS_PER_MM);
    int32_t acc_val   = (int32_t)acc_hz_s;
    int32_t dec_val   = (int32_t)dec_hz_s;
    uint16_t cur_val  = (uint16_t)(current_pct * 10.0f);

    uint16_t pos_l, pos_h, spd_l, spd_h, acc_l, acc_h, dec_l, dec_h;
    to_u32_words(pos_steps, &pos_l, &pos_h);
    to_u32_words(spd_hz,    &spd_l, &spd_h);
    to_u32_words(acc_val,   &acc_l, &acc_h);
    to_u32_words(dec_val,   &dec_l, &dec_h);

    /* Write REG_OP_TYPE (0x0107) through REG_CURRENT (0x0110) — 10 registers */
    uint16_t buf[10] = {
        (uint16_t)op_type,    /* 0x0107 : operation type  */
        pos_l, pos_h,         /* 0x0108-0x0109 : position */
        spd_l, spd_h,         /* 0x010A-0x010B : speed    */
        acc_l, acc_h,         /* 0x010C-0x010D : accel    */
        dec_l, dec_h,         /* 0x010E-0x010F : decel    */
        cur_val,              /* 0x0110 : current         */
    };
    if (write_regs(drv, REG_OP_TYPE, buf, 10) != 0) return -1;

    /* Trigger: assert TRIG bit, then clear (rising-edge detection) */
    write_reg(drv, REG_FIXED_IO_IN, BIT_TRIG);
    sleep_ms(10);
    write_reg(drv, REG_FIXED_IO_IN, 0);

    const char *op_name = (op_type == OP_ABSOLUTE)        ? "ABS"
                        : (op_type == OP_INCREMENTAL_CMD)  ? "INC-CMD"
                                                           : "INC-FB";
    printf("[%s] target=%+.3f mm (%+d steps)  speed=%.1f mm/s\n",
           op_name, position_mm, pos_steps, speed_mm_s);

    if (!wait) return 0;

    int rc = azdkren_wait_in_position(drv, 60.0f);
    if (rc == 0)
        printf("IN-POS reached.  Feedback position: %.3f mm\n",
               azdkren_get_position(drv));
    return rc;
}

int azdkren_move_abs(AZDKREN *drv,
                     float pos_mm,   float speed_mm_s,
                     float acc_hz_s, float dec_hz_s,
                     float current_pct, int wait)
{
    return direct_move(drv, pos_mm, speed_mm_s, acc_hz_s, dec_hz_s,
                       current_pct, OP_ABSOLUTE, wait);
}

int azdkren_move_rel(AZDKREN *drv,
                     float delta_mm, float speed_mm_s,
                     float acc_hz_s, float dec_hz_s,
                     float current_pct, int wait)
{
    return direct_move(drv, delta_mm, speed_mm_s, acc_hz_s, dec_hz_s,
                       current_pct, OP_INCREMENTAL_FB, wait);
}
