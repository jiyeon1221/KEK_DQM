#ifndef TBAstroReader_h
#define TBAstroReader_h 1

// ─────────────────────────────────────────────────────────────────────────────
//  TBAstroReader
//
//  Reader for AstroPix (DAQ "Type C") .dat files. AstroPix data is NOT in the
//  standard TB waveform/fastmode format (TBread/FileController), because the
//  event/hit layout is dictated by the ASIC readout protocol, not by our TCB.
//  This class is deliberately independent from TBread so integrating it does
//  not touch any existing reading path.
//
//  File layout (validated against toy_astroPix/Run_953):
//    - No file-level header; the file starts directly with a 4-byte
//      0xFFFFFFFF event marker.
//    - Each event = 16-byte header + variable-length hit block, and the next
//      event's marker immediately follows (no gap/padding between events).
//    - Header (16 bytes, starting at the marker itself):
//        byte[7]     = trigger fine time
//        byte[8..13] = trigger coarse time (6 bytes, little-endian)
//        triggerTime[ns] = fine*8 + coarse*1000
//    - Hit block (from byte 16 on): 5-byte little-endian hit words, with
//      0xBC idle/padding bytes interleaved and a terminating "0xFF 0xFF"
//      (i.e. the leading two bytes of the NEXT event's marker).
//    - The last marker in a file is only used as an end-of-data boundary
//      (mirrors the reference DAQTypeCParser::findEnd()); the segment after
//      it is never decoded as an event, since it may be truncated mid-write.
//
//  Reference for the decode logic (external repo, not part of this package):
//  DAQTypeCParser.cpp / functions.h (Flip()) — ported here so nothing from
//  that repo is included as a dependency.
//
//  API intentionally mirrors TBread's conventions (multi-file iteration,
//  sequential GetAnEvent()) so a later TBaux integration is a drop-in.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <string>
#include <vector>

struct TBAstroHit {
  unsigned short aid = 0;    // chip id
  unsigned short pay = 0;    // payload
  unsigned short isCol = 0;  // 1 = column hit, 0 = row hit
  unsigned short ch = 0;     // location (0-34)
  unsigned short ts = 0;     // timestamp
  unsigned short tot = 0;    // time over threshold: (tot_msb << 8) + tot_lsb
};

struct TBAstroEvent {
  unsigned long long triggerTime = 0;  // ns, fine*8 + coarse*1000
  std::vector<TBAstroHit> hits;        // already rsv/ch-filtered
};

class TBAstroReader {
public:
  // File path convention (same multi-file layout as TBread's FileController):
  //   <baseDir>/Run_<N>/Run_<N>_Astro/Run_<N>_Astro_MID_<M>/
  //     Run_<N>_Astro_MID_<M>_FILE_<K>.dat   (K = 0, 1, 2, ...)
  TBAstroReader(int runNum, int mid, const std::string& baseDir);

  int GetMaxEvent() const { return fTotalMaxEvent; }
  int GetCurrentEvent() const { return fTotalCurrentEvent; }

  // Sequential read; auto-advances across FILE_K boundaries. Throws
  // std::runtime_error if called past the last available event.
  TBAstroEvent GetAnEvent();

private:
  std::string GetFileName(int fileNum) const;
  void ScanFiles();
  bool LoadFile(int fileIdx);
  static std::vector<size_t> FindMarkers(const std::vector<char>& buf);
  static unsigned short Flip(unsigned short a, int nbit);
  TBAstroEvent DecodeEvent(size_t start, size_t end) const;

  int fRunNum;
  int fMID;
  std::string fBaseDir;

  std::vector<std::string> fFileNames;
  std::vector<int> fFileEventCounts;
  int fTotalMaxEvent;
  int fTotalCurrentEvent;

  int fCurrentFileIdx;
  std::vector<char> fBuffer;
  std::vector<size_t> fMarkers;   // all marker offsets found in fBuffer
  size_t fEventCursorInFile;      // index into fMarkers for the next event
};

#endif
