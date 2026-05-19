#ifndef REGISTERS_H
#define REGISTERS_H

/* ============================================================
 *  registers.h
 *  AZD-KREN Ethernet Driver — Modbus register map
 *  Reference: minidriver_AZD_KREN.pdf
 * ============================================================ */

/* ── Output registers (PC → Driver, Holding Registers) ────────────────── */
#define REG_REMOTE_IO_IN    0x0104   /* 260  Remote I/O R-IN                */
#define REG_DATA_NO_SEL     0x0105   /* 261  Operation data number selection */
#define REG_FIXED_IO_IN     0x0106   /* 262  Fixed I/O IN  (bit field)       */
#define REG_OP_TYPE         0x0107   /* 263  Direct data operation type      */
#define REG_POS_L           0x0108   /* 264  Position lower word  (steps)    */
#define REG_POS_H           0x0109   /* 265  Position upper word             */
#define REG_SPD_L           0x010A   /* 266  Speed lower word    (Hz)        */
#define REG_SPD_H           0x010B   /* 267  Speed upper word                */
#define REG_ACC_L           0x010C   /* 268  Acceleration rate lower (Hz/s)  */
#define REG_ACC_H           0x010D   /* 269  Acceleration rate upper         */
#define REG_DEC_L           0x010E   /* 270  Deceleration rate lower (Hz/s)  */
#define REG_DEC_H           0x010F   /* 271  Deceleration rate upper         */
#define REG_CURRENT         0x0110   /* 272  Operating current (0-1000, 0.1%)*/
#define REG_FWD_DEST        0x0111   /* 273  Forwarding dest (0=exec,1=buf)  */
#define REG_WR_REQ          0x0114   /* 276  Write request (bit0=WR-REQ)     */
#define REG_WR_PARAM_ID     0x0115   /* 277  Write parameter ID              */
#define REG_WR_DATA_L       0x0116   /* 278  Write data lower word           */
#define REG_WR_DATA_H       0x0117   /* 279  Write data upper word           */

/* ── Input registers (Driver → PC) ────────────────────────────────────── */
#define REG_REMOTE_IO_OUT   0x011C   /* 284  Remote I/O R-OUT               */
#define REG_FIXED_IO_OUT    0x011E   /* 286  Fixed I/O OUT (bit field)       */
#define REG_ALARM_CODE      0x011F   /* 287  Present alarm code              */
#define REG_FDBK_POS_L      0x0120   /* 288  Feedback position lower (steps) */
#define REG_FDBK_POS_H      0x0121   /* 289  Feedback position upper         */
#define REG_FDBK_SPD_L      0x0122   /* 290  Feedback speed lower  (Hz)      */
#define REG_FDBK_SPD_H      0x0123   /* 291  Feedback speed upper            */
#define REG_RW_STATUS       0x012C   /* 300  R/W status (bit8=WR-END)        */

/* ── Maintenance command registers ─────────────────────────────────────── */
#define REG_P_PRESET        0x008B   /* Position preset: write 1 → current pos = 0 */
#define REG_NVM_WRITE       0x0192   /* NV-memory save: write 1 → persist to flash */

/* ── Fixed I/O IN bit masks  (REG_FIXED_IO_IN, 0x0106) ─────────────────── */
#define BIT_START           (1 << 3)   /* START signal                */
#define BIT_HOME            (1 << 4)   /* HOME / origin-return        */
#define BIT_STOP            (1 << 5)   /* STOP signal                 */
#define BIT_FREE            (1 << 6)   /* FREE — motor torque off     */
#define BIT_ALM_RST         (1 << 7)   /* ALM-RST — alarm reset       */
#define BIT_TRIG            (1 << 8)   /* TRIG — direct data trigger  */

/* ── Fixed I/O OUT bit masks (REG_FIXED_IO_OUT, 0x011E) ────────────────── */
#define BIT_IN_POS          (1 << 2)   /* IN-POS  — reached target    */
#define BIT_READY           (1 << 5)   /* READY   — driver ready      */
#define BIT_DCMD_RDY        (1 << 6)   /* DCMD-RDY — direct cmd ready */
#define BIT_ALM_A           (1 << 7)   /* ALM-A   — alarm active      */

/* ── Direct data operation types (REG_OP_TYPE, 0x0107) ─────────────────── */
#define OP_ABSOLUTE         1    /* absolute position              */
#define OP_INCREMENTAL_CMD  2    /* incremental from command pos   */
#define OP_INCREMENTAL_FB   3    /* incremental from feedback pos  */

/* ── Motor / driver physical constants ─────────────────────────────────── */
/*
 * EZSM6020AZAK (EZS6 series, 20 mm stroke, AZ motor built-in):
 *   Lead screw pitch    : 8 mm / rev   (EZS6 standard)
 *   Driver resolution   : 1000 steps / rev  (AZ Series default)
 *   → STEPS_PER_MM      = 1000 / 8 = 125 steps/mm
 *   → Stroke max        : 20 mm → 2500 steps
 *
 * Verify these values against your MEXE02 parameter settings.
 */
#define STEPS_PER_MM        100.0f   /* 0.01 mm/step — confirmed in MEXE02 title bar */
#define STROKE_MAX_MM       200.0f   /* EZSM6E020AZAK: 200 mm stroke, 6 mm/rev lead */

#endif /* REGISTERS_H */
