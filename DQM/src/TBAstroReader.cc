#include "TBAstroReader.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifndef _WIN32
#include <unistd.h>  // access
#endif

TBAstroReader::TBAstroReader(int runNum, int mid, const std::string& baseDir)
    : fRunNum(runNum),
      fMID(mid),
      fBaseDir(baseDir),
      fTotalMaxEvent(0),
      fTotalCurrentEvent(0),
      fCurrentFileIdx(-1),
      fEventCursorInFile(0) {
  ScanFiles();
}

std::string TBAstroReader::GetFileName(int fileNum) const {
  return fBaseDir + "/Run_" + std::to_string(fRunNum) + "/Run_" + std::to_string(fRunNum) +
         "_Astro/Run_" + std::to_string(fRunNum) + "_Astro_MID_" + std::to_string(fMID) +
         "/Run_" + std::to_string(fRunNum) + "_Astro_MID_" + std::to_string(fMID) + "_FILE_" +
         std::to_string(fileNum) + ".dat";
}

std::vector<size_t> TBAstroReader::FindMarkers(const std::vector<char>& buf) {
  std::vector<size_t> markers;
  if (buf.size() < 4) return markers;

  for (size_t i = 0; i + 4 <= buf.size(); ++i) {
    if (static_cast<unsigned char>(buf[i]) == 0xFF &&
        static_cast<unsigned char>(buf[i + 1]) == 0xFF &&
        static_cast<unsigned char>(buf[i + 2]) == 0xFF &&
        static_cast<unsigned char>(buf[i + 3]) == 0xFF) {
      markers.push_back(i);
    }
  }
  return markers;
}

void TBAstroReader::ScanFiles() {
  fFileNames.clear();
  fFileEventCounts.clear();
  fTotalMaxEvent = 0;

  for (int fileNum = 0; fileNum < 999; ++fileNum) {
    std::string fileName = GetFileName(fileNum);
    if (access(fileName.c_str(), F_OK)) break;

    std::ifstream in(fileName, std::ios::binary);
    if (!in) break;

    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    if (size <= 0) {
      // Empty file: no more events, but do not stop the scan — a live DAQ
      // may still be writing later FILE_K, so just record 0 events for this
      // one and keep looking (matches TBread's tolerant multi-file scan).
      fFileNames.push_back(fileName);
      fFileEventCounts.push_back(0);
      continue;
    }
    in.seekg(0, std::ios::beg);

    std::vector<char> buf(static_cast<size_t>(size));
    in.read(buf.data(), size);

    std::vector<size_t> markers = FindMarkers(buf);
    int nEvent = (markers.size() >= 1) ? static_cast<int>(markers.size()) - 1 : 0;

    fFileNames.push_back(fileName);
    fFileEventCounts.push_back(nEvent);
    fTotalMaxEvent += nEvent;

    std::cout << "[TBAstroReader] file scanning : " << fileName << " - Max Event : " << nEvent
              << " / " << fTotalMaxEvent << std::endl;
  }

  if (fFileNames.empty())
    throw std::runtime_error("TBAstroReader::ScanFiles - no files found for Run " +
                              std::to_string(fRunNum) + " MID " + std::to_string(fMID) +
                              " (looked for " + GetFileName(0) + ")");
}

bool TBAstroReader::LoadFile(int fileIdx) {
  if (fileIdx < 0 || fileIdx >= static_cast<int>(fFileNames.size())) return false;

  std::ifstream in(fFileNames[fileIdx], std::ios::binary);
  if (!in) return false;

  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);

  fBuffer.assign(static_cast<size_t>(size > 0 ? size : 0), 0);
  if (size > 0) in.read(fBuffer.data(), size);

  fMarkers = FindMarkers(fBuffer);
  fEventCursorInFile = 0;
  fCurrentFileIdx = fileIdx;

  return true;
}

unsigned short TBAstroReader::Flip(unsigned short a, int nbit) {
  unsigned short b = 0;
  for (int i = 0; i < nbit; ++i) b += (((a >> i) & 0x1) << (nbit - 1 - i));
  return b;
}

TBAstroEvent TBAstroReader::DecodeEvent(size_t start, size_t end) const {
  TBAstroEvent evt;
  if (end < start || end - start < 16) return evt;

  const char* data = fBuffer.data() + start;
  const size_t len = end - start;

  const int fine = static_cast<unsigned char>(data[7]) & 0xFF;
  unsigned long long coarse = 0;
  for (int a = 0; a < 6; ++a)
    coarse += (static_cast<unsigned long long>(static_cast<unsigned char>(data[8 + a]) & 0xFF)
               << (8 * a));
  evt.triggerTime = static_cast<unsigned long long>(fine) * 8ULL + coarse * 1000ULL;

  size_t idx = 16;
  uint8_t tmp = 0x00;

  while (idx + 1 < len) {
    const uint8_t value = static_cast<uint8_t>(data[idx]);
    const uint8_t valueNext = static_cast<uint8_t>(data[idx + 1]);
    bool flag = true;

    if (value == 0xBC) {
      if (tmp == 0xBC || valueNext == 0xBC) flag = false;
    } else if (value == 0xFF && valueNext == 0xFF) {
      break;  // leading two bytes of the next event's marker
    }

    if (flag) {
      if (idx + 5 > len) break;  // malformed/truncated tail, stop decoding

      const unsigned long word =
          static_cast<unsigned long>(static_cast<uint8_t>(data[idx])) |
          (static_cast<unsigned long>(static_cast<uint8_t>(data[idx + 1])) << 8) |
          (static_cast<unsigned long>(static_cast<uint8_t>(data[idx + 2])) << 16) |
          (static_cast<unsigned long>(static_cast<uint8_t>(data[idx + 3])) << 24) |
          (static_cast<unsigned long>(static_cast<uint8_t>(data[idx + 4])) << 32);
      idx += 5;

      const unsigned short aid = (word & 0x000000001FULL) >> 0;
      const unsigned short pay = (word & 0x00000000E0ULL) >> 5;
      const unsigned short isCol = (word & 0x0000000100ULL) >> 8;
      const unsigned short rsv1 = (word & 0x0000000200ULL) >> 9;
      const unsigned short ch = (word & 0x000000FC00ULL) >> 10;
      const unsigned short ts = (word & 0x0000FF0000ULL) >> 16;
      const unsigned short rsv2 = (word & 0x000F000000ULL) >> 24;
      const unsigned short msb = (word & 0x00F0000000ULL) >> 28;
      const unsigned short lsb = (word & 0xFF00000000ULL) >> 32;

      const unsigned short faid = Flip(aid, 5);
      const unsigned short fpay = Flip(pay, 3);
      const unsigned short fch = Flip(ch, 6);
      const unsigned short fts = Flip(ts, 8);
      const unsigned short fmsb = Flip(msb, 4);
      const unsigned short flsb = Flip(lsb, 8);

      const unsigned short tot = flsb + static_cast<unsigned short>(fmsb << 8);

      if (rsv1 != 0 || rsv2 != 0) continue;
      if (fch > 34) continue;

      TBAstroHit hit;
      hit.aid = faid;
      hit.pay = fpay;
      hit.isCol = isCol;
      hit.ch = fch;
      hit.ts = fts;
      hit.tot = tot;
      evt.hits.push_back(hit);
    } else {
      idx += 1;
      tmp = value;
    }
  }

  return evt;
}

TBAstroEvent TBAstroReader::GetAnEvent() {
  if (fCurrentFileIdx < 0) {
    if (!LoadFile(0))
      throw std::runtime_error("TBAstroReader::GetAnEvent - could not open first file");
  }

  while (fMarkers.size() < 2 || fEventCursorInFile + 1 >= fMarkers.size()) {
    // Current file exhausted (or has no usable events); move to the next one.
    const int nextIdx = fCurrentFileIdx + 1;
    if (!LoadFile(nextIdx))
      throw std::runtime_error(
          "TBAstroReader::GetAnEvent - no more events (requested past end of available data)");
  }

  const size_t start = fMarkers[fEventCursorInFile];
  const size_t end = fMarkers[fEventCursorInFile + 1];
  TBAstroEvent evt = DecodeEvent(start, end);

  ++fEventCursorInFile;
  ++fTotalCurrentEvent;

  return evt;
}
