
#ifndef WAVEPACKETS_ON_SINGLE_DEVICE
#define WAVEPACKETS_ON_SINGLE_DEVICE

class WavepacketsOnSingleDevice
{
public:
  WavepacketsOnSingleDevice(const int device_index,
			    const int omega_start,
			    const int n_omegas);

  ~WavepacketsOnSingleDevice() { destroy_data_on_device(); }

private:

  int _device_index;
  int omega_start;
  int n_omegas;

  int device_index() const { return _device_index; }
  int current_device_index() const;
  void setup_device() const;

  double *potential_dev;

  void setup_data_on_device();
  void destroy_data_on_device();

  void setup_potential_on_device();

};

#endif /* WAVEPACKETS_ON_SINGLE_DEVICE */
