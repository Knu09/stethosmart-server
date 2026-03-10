#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct wav_header {
  char riff[4];
  int32_t flength;
  char wave[4];
  char fmt[4];
  int32_t chunk_size;
  int16_t format_tag;
  int16_t num_chans;
  int32_t sample_rate;
  int32_t bytes_per_second;
  int16_t bytes_per_sample;
  int16_t bits_per_sample;
  char data[4];
  int32_t dlength;
};

int main() {
  struct wav_header wavh;

  strncpy(wavh.riff, "RIFF", 4);
  strncpy(wavh.wave, "WAVE", 4);
  strncpy(wavh.fmt, "fmt ", 4);
  strncpy(wavh.data, "data", 4);

  wavh.chunk_size = 16;
  wavh.format_tag = 1;
  wavh.num_chans = 1;
  wavh.sample_rate = 16000;
  wavh.bits_per_sample = 16;
  wavh.bytes_per_sample = (wavh.bits_per_sample / 8) * wavh.num_chans;
  wavh.bytes_per_second = wavh.sample_rate * wavh.bytes_per_sample;

  int sample_rate = wavh.sample_rate;

  const int duration = 20;
  const int buffer_size = sample_rate * duration;

  wavh.dlength = buffer_size * wavh.bytes_per_sample;
  wavh.flength = wavh.dlength + 36;

  FILE *in = fopen("pcm_data.bin", "rb");
  FILE *out = fopen("lung.wav", "wb");

  fwrite(&wavh, 1, sizeof(wavh), out);
  unsigned char buffer[320000];
  size_t bytes_read;
  while ((bytes_read = fread(buffer, 1, sizeof(buffer), in)) > 0) {
    fwrite(buffer, 1, bytes_read, out);
  }

  fclose(in);
  fclose(out);

  return 0;
}
