/* ============================================================
 *  main.c — azd_kren command-line interface
 *
 *  Build:  make
 *
 *  Quick reference:
 *    ./azd_kren --status
 *    ./azd_kren --home
 *    ./azd_kren --moveto 10.5
 *    ./azd_kren --move -3.0 --speed 5.0
 *    ./azd_kren --sethome
 *    ./azd_kren --stop
 *    ./azd_kren --ip 192.168.1.3 --moveto 15.0
 *
 *  Home offset:
 *    --sethome saves the current raw position to ~/.azd_kren_home.
 *    All subsequent --moveto / --pos commands are expressed relative
 *    to that saved origin.  --home (origin-return) clears the offset.
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "azd_kren.h"

/* ── Home-offset persistence ────────────────────────────────────────────── */

static const char *home_file_path(void)
{
    static char path[512];
    const char *home = getenv("HOME");
    snprintf(path, sizeof(path), "%s/.azd_kren_home", home ? home : "/tmp");
    return path;
}

static float load_home_offset(void)
{
    FILE *f = fopen(home_file_path(), "r");
    if (!f) return 0.0f;
    float offset = 0.0f;
    fscanf(f, "%f", &offset);
    fclose(f);
    return offset;
}

static void save_home_offset(float raw_mm)
{
    FILE *f = fopen(home_file_path(), "w");
    if (!f) { fprintf(stderr, "Warning: cannot save home offset to %s\n", home_file_path()); return; }
    fprintf(f, "%.4f\n", raw_mm);
    fclose(f);
}

/* ── Command enum ───────────────────────────────────────────────────────── */
typedef enum {
    CMD_NONE,
    CMD_STATUS,
    CMD_POS,
    CMD_HOME,
    CMD_SETHOME,
    CMD_MOVETO,
    CMD_MOVE,
    CMD_STOP,
    CMD_FREE,
    CMD_ON,
    CMD_ALARM_RESET,
} Command;

/* ── Usage ──────────────────────────────────────────────────────────────── */
static void print_usage(const char *prog)
{
    printf(
        "\nUsage:\n"
        "  %s [global opts] <command> [command opts]\n"
        "\n"
        "Global options:\n"
        "  --ip    <addr>   Driver IP address       (default: 192.168.1.1)\n"
        "  --port  <n>      Modbus TCP port          (default: 502)\n"
        "  --slave <n>      Modbus slave ID          (default: 0, AZD-KREN Ethernet)\n"
        "\n"
        "Commands:\n"
        "  --status         Print driver status and position relative to home\n"
        "  --pos            Print current position relative to home [mm]\n"
        "  --home           Execute origin return (motor moves to sensor)\n"
        "                   Also clears any --sethome offset\n"
        "  --sethome        Declare current position as coordinate origin\n"
        "                   (motor does NOT move; offset saved to ~/.azd_kren_home)\n"
        "  --moveto <mm>    Absolute move to position relative to home [mm]\n"
        "  --move   <mm>    Relative move by offset  [mm]  (+ / - allowed)\n"
        "  --stop           Decelerate and stop\n"
        "  --free           Disable motor torque (motor can be pushed by hand)\n"
        "  --on             Re-enable motor torque\n"
        "  --alarm-reset    Reset active alarm\n"
        "\n"
        "Motion options  (for --moveto and --move):\n"
        "  --speed   <mm/s> Travel speed             (default: 20.0)\n"
        "  --acc     <Hz/s> Acceleration rate         (default: 100000)\n"
        "  --dec     <Hz/s> Deceleration rate         (default: 100000)\n"
        "  --current <%%>    Motor operating current   (default: 100.0)\n"
        "  --nowait         Return immediately; do not wait for IN-POS\n"
        "\n"
        "Examples:\n"
        "  %s --status\n"
        "  %s --home\n"
        "  %s --moveto 10.5\n"
        "  %s --moveto 10.5 --speed 5.0 --acc 50000 --dec 50000\n"
        "  %s --move -3.0\n"
        "  %s --sethome\n"
        "  %s --stop\n"
        "  %s --alarm-reset\n"
        "  %s --ip 192.168.1.2 --moveto 15.0\n"
        "\n",
        prog,
        prog, prog, prog, prog, prog, prog, prog, prog, prog
    );
}

/* ── Argument helpers ───────────────────────────────────────────────────── */

static const char *next_arg(int *i, int argc, char **argv, const char *opt)
{
    (*i)++;
    if (*i >= argc) {
        fprintf(stderr, "Error: %s requires an argument.\n", opt);
        exit(1);
    }
    return argv[*i];
}

static float parse_float(const char *s, const char *opt)
{
    char *end;
    float v = strtof(s, &end);
    if (*end != '\0') {
        fprintf(stderr, "Error: invalid number '%s' for %s.\n", s, opt);
        exit(1);
    }
    return v;
}

static int parse_int(const char *s, const char *opt)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (*end != '\0') {
        fprintf(stderr, "Error: invalid integer '%s' for %s.\n", s, opt);
        exit(1);
    }
    return (int)v;
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* ── Defaults ──────────────────────────────────────────────────────── */
    const char *ip        = "192.168.1.1";
    int         port      = 502;
    int         slave     = 0;
    Command     cmd       = CMD_NONE;
    float       target_mm = 0.0f;
    float       speed     = 20.0f;
    float       acc       = 100000.0f;
    float       dec       = 100000.0f;
    float       current   = 100.0f;
    int         nowait    = 0;

    /* ── Parse arguments ───────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];

        if      (strcmp(a, "--ip")    == 0) { ip    = next_arg(&i, argc, argv, a); }
        else if (strcmp(a, "--port")  == 0) { port  = parse_int(next_arg(&i, argc, argv, a), a); }
        else if (strcmp(a, "--slave") == 0) { slave = parse_int(next_arg(&i, argc, argv, a), a); }

        else if (strcmp(a, "--status")      == 0) { cmd = CMD_STATUS; }
        else if (strcmp(a, "--pos")         == 0) { cmd = CMD_POS; }
        else if (strcmp(a, "--home")        == 0) { cmd = CMD_HOME; }
        else if (strcmp(a, "--sethome")     == 0) { cmd = CMD_SETHOME; }
        else if (strcmp(a, "--stop")        == 0) { cmd = CMD_STOP; }
        else if (strcmp(a, "--free")        == 0) { cmd = CMD_FREE; }
        else if (strcmp(a, "--on")          == 0) { cmd = CMD_ON; }
        else if (strcmp(a, "--alarm-reset") == 0) { cmd = CMD_ALARM_RESET; }
        else if (strcmp(a, "--moveto") == 0) {
            cmd = CMD_MOVETO;
            target_mm = parse_float(next_arg(&i, argc, argv, a), a);
        }
        else if (strcmp(a, "--move") == 0) {
            cmd = CMD_MOVE;
            target_mm = parse_float(next_arg(&i, argc, argv, a), a);
        }

        else if (strcmp(a, "--speed")   == 0) { speed   = parse_float(next_arg(&i, argc, argv, a), a); }
        else if (strcmp(a, "--acc")     == 0) { acc     = parse_float(next_arg(&i, argc, argv, a), a); }
        else if (strcmp(a, "--dec")     == 0) { dec     = parse_float(next_arg(&i, argc, argv, a), a); }
        else if (strcmp(a, "--current") == 0) { current = parse_float(next_arg(&i, argc, argv, a), a); }
        else if (strcmp(a, "--nowait")  == 0) { nowait  = 1; }

        else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "Unknown option: %s  (use --help)\n", a);
            return 1;
        }
    }

    if (cmd == CMD_NONE) {
        fprintf(stderr, "No command specified. Use --help.\n");
        return 1;
    }

    /* ── Load home offset ──────────────────────────────────────────────── */
    float home_offset = load_home_offset();   /* raw driver mm at user's home */

    /* ── Connect ───────────────────────────────────────────────────────── */
    AZDKREN drv;
    if (azdkren_connect(&drv, ip, port, slave) != 0)
        return 1;

    int ret = 0;

    /* ── Dispatch ──────────────────────────────────────────────────────── */
    switch (cmd) {

    case CMD_STATUS: {
        AZDStatus st;
        if (azdkren_get_status(&drv, &st) != 0) { ret = 1; break; }
        float raw_pos = azdkren_get_position(&drv);
        printf("Status → ready=%d  in_pos=%d  dcmd_rdy=%d  alarm=%d",
               st.ready, st.in_pos, st.dcmd_rdy, st.alarm);
        if (st.alarm)
            printf("  (alarm_code=0x%04X)", st.alarm_code);
        if (raw_pos >= 0.0f)
            printf("  position=%.3f mm", raw_pos - home_offset);
        printf("\n");
        break;
    }

    case CMD_POS: {
        float raw_pos = azdkren_get_position(&drv);
        if (raw_pos < 0.0f) { ret = 1; break; }
        printf("%.3f mm\n", raw_pos - home_offset);
        break;
    }

    case CMD_HOME:
        ret = azdkren_home(&drv, /*wait=*/1);
        if (ret == 0) {
            save_home_offset(0.0f);   /* physical home = raw 0 → reset offset */
            printf("Home offset cleared.\n");
        }
        break;

    case CMD_SETHOME: {
        float raw_pos = azdkren_get_position(&drv);
        if (raw_pos < 0.0f) { ret = 1; break; }
        save_home_offset(raw_pos);
        printf("Home set here.  Raw position %.3f mm is now 0.000 mm.\n", raw_pos);
        break;
    }

    case CMD_MOVETO:
        /* target_mm is relative to user's home → add offset for driver */
        ret = azdkren_move_abs(&drv, target_mm + home_offset,
                               speed, acc, dec, current, !nowait);
        break;

    case CMD_MOVE:
        ret = azdkren_move_rel(&drv, target_mm, speed, acc, dec, current, !nowait);
        break;

    case CMD_STOP:
        ret = azdkren_stop(&drv);
        break;

    case CMD_FREE:
        ret = azdkren_motor_free(&drv);
        break;

    case CMD_ON:
        ret = azdkren_motor_on(&drv);
        break;

    case CMD_ALARM_RESET:
        ret = azdkren_reset_alarm(&drv);
        break;

    default:
        break;
    }

    azdkren_disconnect(&drv);
    return (ret == 0) ? 0 : 1;
}
