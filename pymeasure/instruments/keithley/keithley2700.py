#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2017 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import logging
import numpy as np
from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import truncated_range, truncated_discrete_set, strict_discrete_set
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# TODO:
"""
1. filter
2. rate and bw
3. save and recall configs
4. triggering
5. implement binary transfer parsing
"""


class KeithleyData(object):
    """
    Helper class for parsing data transfers from Keithley 2700 MultiMeter.

    .. code-block:: python

        from pymeasure.instruments.keithley import Keithley2700, KeithleyData

        meter = Keithley2700('COM1')
        meter.mode = 'voltage'
        meter.configure_data_format(timestamp=True, reading_number=True, units=True)
        data = KeithleyData(meter.multi_point_measurement(samples=10), meter.data_elements, meter.data_format)
        print(str(data))
        print(data.readings)

    """
    UNITS = ['ADC', 'AAC', 'VDC', 'VAC', 'SECS', 'HZ', 'SEC', 'RDNG#', 'LIMITS']

    def __init__(self, data_string, elements, format_string='ascii', byte_order='normal'):
        print(elements)
        self.data_string = data_string  # reading,timestamp,reading_number,channel,limit (if all included)
        self.elements = elements
        self.format_string = format_string
        self.byte_order = byte_order

        self.readings = None
        self.timestamps = None
        self.reading_numbers = None
        self.channels = None
        self.limits = None
        self.limits_high2 = None
        self.limits_low2 = None
        self.limits_high1 = None
        self.limits_low1 = None

        if self.format_string == 'ascii':
            # ASCII transfer format
            count = len(self.elements)
            if 'units' in self.elements:
                # parse out units from data string
                count -= 1
                for unit in self.UNITS:
                    data_string = data_string.replace(unit, '')

            self.data = data_string.split(',')

            start = 0
            # take every 'count' element from list starting at 'start' and convert to the appropriate data type
            if 'readings' in self.elements:
                self.readings = np.fromstring(','.join(self.data[start::count]), sep=',', dtype=np.float)
                start += 1
            if 'timestamp' in self.elements:
                self.timestamps = np.array(self.data[start::count], dtype=np.float)
                start += 1
            if 'reading_number' in self.elements:
                self.reading_numbers = np.array(self.data[start::count], dtype=np.int)
                start += 1
            if 'channel' in self.elements:
                self.channels = np.array(self.data[start::count], dtype=np.str)
                start += 1
            if 'limits' in self.elements:
                self.limits = np.array(self.data[start::count], dtype=np.byte)
                self.limits_high2 = np.array(self.limits & 0x01, dtype=np.bool)
                self.limits_low2 = np.array(self.limits & 0x02, dtype=np.bool)
                self.limits_high1 = np.array(self.limits & 0x04, dtype=np.bool)
                self.limits_low1 = np.array(self.limits & 0x08, dtype=np.bool)
                start += 1
        else:
            # binary transfer format
            # TODO: implement
            raise NotImplementedError("Binary data formats not currently supported")

    def __str__(self):
        return (
            'Readings: {}\n'
            'Timestamps: {}\n'
            'Reading Numbers: {}\n'
            'Channel Numbers: {}\n'
            'High Limit 2: {}\n'
            'Low Limit 2: {}\n'
            'High Limit 1: {}\n'
            'Low Limit 1: {}'.format(self.readings, self.timestamps, self.reading_numbers, self.channels,
                                     self.limits_high2, self.limits_low2, self.limits_high1, self.limits_low1)
        )

    def mean(self):
        if self.readings is not None:
            return np.mean(self.readings)


class Keithley2700(Instrument):
    """
    Represents the Keithley 2700 MultiMeter System and provides a
    high-level interface for interacting with the instrument.
    NOTE: Binary transfer data conversion is currently not supported.

    .. code-block:: python

        from pymeasure.instruments.keithley import Keithley2700, KeithleyData

        meter = Keithley2700('COM1')
        meter.mode = 'voltage'
        meter.configure_data_format(timestamp=True, reading_number=True, units=True)
        data = KeithleyData(meter.multi_point_measurement(samples=10), meter.data_elements, meter.data_format)
        print(str(data))
        print(data.readings)

    """
    MODES = {
        'current': 'CURR:DC', 'current ac': 'CURR:AC',
        'voltage': 'VOLT:DC', 'voltage ac': 'VOLT:AC',
        'resistance': 'RES', 'resistance 4W': 'FRES',
        'period': 'PER', 'frequency': 'FREQ',
        'temperature': 'TEMP', 'continuity': 'CONT'
    }

    RANGES = {
        'current': [0.02, 0.1, 1, 3], 'current ac': [1, 3],
        'voltage': [0.1, 1, 10, 100, 1000], 'voltage ac': [0.1, 1, 10, 100, 750],
        'resistance': [1e2, 1e3, 1e4, 1e5, 1e5, 1e6, 1e7, 1e8],
        'resistance 4W': [1e2, 1e3, 1e4, 1e5, 1e5, 1e6, 1e7, 1e8],
    }

    DIGITS = {
        'current': [4, 7], 'current ac': [4, 7],
        'voltage': [4, 7], 'voltage ac': [4, 7],
        'resistance': [4, 7], 'resistance 4W': [4, 7],
        'frequency': [4, 7], 'period': [4, 7],
        'temperature': [4, 7]
    }

    DATA_FORMAT = {
        'ascii': 'ASC', 'single': 'SRE', 'double': 'DRE'
    }

    DATA_ELEMENTS = {
        'readings': 'READ', 'timestamp': 'TST', 'units': 'UNIT',
        'reading_number': 'RNUM', 'channel': 'CHAN', 'limits': 'LIM'
    }

    BYTE_ORDER = {
        'normal': 'NORM', 'swapped': 'SWAP'
    }

    mode = Instrument.control(
        "SENS:FUNC?", "SENS:FUNC '%s'",
        """A string property that controls the configuration mode for measurements,
        which can take the values: :attr:'current' (DC), :attr:'current ac', 
        :attr:'voltage' (DC),  :attr:'voltage ac', :attr:'resistance' (2-wire), 
        :attr:'resistance 4W' (4-wire), :attr:'period', :attr:'frequency', 
        :attr:'temperature', and :attr:'continuity'.""",
        validator=strict_discrete_set,
        values=MODES,
        map_values=True,
        get_process=lambda v: v.replace('"', '')
    )

    data_format = Instrument.control(
        "FORMat:DATA?", "FORMat:DATA '%s'",
        """A string property that controls the data transfer format, which can take the values: :attr:'ascii' (for 
        ASCII text-based transfer), :attr:'single' (for IEEE-754 single precision format) or :attr:'double' (for
        IEEE-754 double precision format).""",
        validator=strict_discrete_set,
        values=DATA_FORMAT,
        map_values=True,
    )

    byte_order = Instrument.control(
        "FORMat:BORDer?", "FORMat:BORDer %s",
        """A string property that controls the data transfer byte order, which can take the values: :attr:'normal' or 
        :attr:'swapped'.""",
        validator=strict_discrete_set,
        values=BYTE_ORDER,
        map_values=True
    )

    def __init__(self, adapter, **kwargs):
        super(Keithley2700, self).__init__(adapter, "Keithley 2700 MultiMeter", **kwargs)

    def _mode_command(self, mode=None):
        if mode is None:
            mode = self.mode
        return self.MODES[mode]

    @property
    def data_elements(self):
        """
        Reads the data elements used for data transfers. The following are possible elements: :attr:'readings',
        :attr:'timestamp', :attr:'units': :attr:'reading_number', :attr:'channel', :attr:'limits'.
        :return: list containing all data transfer elements
        """
        element_string = self.ask('FORMat:ELEMents?')
        elements = list(filter(lambda x: x != '', element_string.split(',')))
        inverse = {v: k for k, v in self.DATA_ELEMENTS.items()}
        return [inverse[x] for x in elements]

    @data_elements.setter
    def data_elements(self, elements):
        """
        Writes the data elements used for data transfers. The following are possible elements: :attr:'readings',
        :attr:'timestamp', :attr:'units': :attr:'reading_number', :attr:'channel', :attr:'limits'.
        :param elements: list or tuple of strings containing desired data elements to include in data transfers
        :return: None
        """
        elements2 = []
        for element in elements:
            elements2.append(self.DATA_ELEMENTS[strict_discrete_set(element, self.DATA_ELEMENTS)])
        self.write('FORMat:ELEMents {}'.format(','.join(elements2)))

    @property
    def range(self):
        """
        Reads the range setting for the current mode. This property is not valid for the :attr:'period',
        :attr:'frequency', :attr:'continuity', and :attr:'temperature' modes and will raise a ValueError if called
        when in these modes.
        :return: float corresponding to the current range
        """
        mode = self.mode
        if mode in self.RANGES:
            return float(self.ask('{}:RANGe?'.format(self._mode_command(None))))
        else:
            raise ValueError("Range cannot be read in mode '{}'".format(mode))

    @range.setter
    def range(self, value):
        """
        Writes the range setting for the current mode. This property is not valid for the :attr:'period',
        :attr:'frequency', :attr:'continuity', and :attr:'temperature' modes and will raise a ValueError if called
        when in these modes.
        :param value: range to apply
        :return: None
        """
        mode = self.mode
        if mode in self.RANGES:
            self.write('{}:RANGe {}'.format(self._mode_command(mode),
                                            truncated_discrete_set(value, self.RANGES[mode])))
        else:
            raise ValueError("Range cannot be set in mode '{}'".format(mode))

    @property
    def auto_range(self):
        """
        Reads the auto range setting for the active mode. This property is not valid for the :attr:'period',
        :attr:'frequency', :attr:'continuity', and :attr:'temperature' modes and will raise a ValueError if called
        when in these modes.
        :return: boolean corresponding to current auto range setting
        """
        mode = self.mode
        if mode in self.RANGES:
            return bool(int(self.ask('{}:RANGe:AUTO?'.format(self._mode_command(None)))))
        else:
            raise ValueError("Auto range cannot be read in mode '{}'".format(mode))

    @auto_range.setter
    def auto_range(self, enable):
        """
        Writes the auto range setting for the active mode. This property is not valid for the :attr:'period',
        :attr:'frequency', :attr:'continuity', and :attr:'temperature' modes and will raise a ValueError if called
        when in these modes.
        :param enable: boolean corresponding to desired auto range setting
        :return: None
        """
        mode = self.mode
        if mode in self.RANGES:
            self.write('{}:RANGe:AUTO {}'.format(self._mode_command(None),
                                                 truncated_discrete_set(int(enable), [0, 1])))
        else:
            raise ValueError("Auto range cannot be set in mode '{}'".format(mode))

    @property
    def digits(self):
        """
        Reads the range setting for the active mode. This property is not valid for the :attr:'continuity' mode
        and will raise a ValueError if called when in this mode.
        :return: float corresponding to the current digits setting
        """
        mode = self.mode
        if mode in self.DIGITS:
            return float(self.ask('{}:DIGits?'.format(self._mode_command(None))))
        else:
            raise ValueError("Digits cannot be read in mode '{}'".format(mode))

    @digits.setter
    def digits(self, value):
        """
        Writes the range setting for the active mode. This property is not valid for the :attr:'continuity' mode and
        will raise a ValueError if called when in this mode.
        :param value: digits setting value to apply (between 4 and 7)
        :return: None
        """
        mode = self.mode
        if mode in self.RANGES:
            self.write('{}:DIGits {}'.format(self._mode_command(mode),
                                             truncated_range(value, self.DIGITS[mode])))
        else:
            raise ValueError("Digits cannot be set in mode '{}'".format(mode))

    def configure_data_format(self, data_format='ascii', readings=True, timestamp=False, reading_number=False,
                              channel=False, limits=False, units=False, byte_order_normal=True):
        """
        Configures the data transfer format of the instrument. Using this function is equivalent to calling the
        following properties: data_format, data_elements, byte_order
        :param data_format: specifies the desired data transfer format (:attr:'ascii', :attr:'single' or :attr:'double')
        :param readings: boolean for the inclusion of readings in the data return
        :param timestamp: boolean for the inclusion of timestamps in the data return
        :param reading_number: boolean for the inclusion of reading numbers in the data return
        :param channel: boolean for the inclusion of channel strings in the data return
        :param limits: boolean for the inclusion of limits in the data return
        :param units: boolean for the inclusion of units in the data return
        :param byte_order_normal: boolean for specifying the byte order as normal (True) or swapped (False)
        :return: None
        """
        elements = []
        if readings:
            elements.append('READing')
        if units:
            elements.append('UNITs')
        if timestamp:
            elements.append('TSTamp')
        if reading_number:
            elements.append('RNUMber')
        if channel:
            elements.append('CHANnel')
        if limits:
            elements.append('LIMits')
        self.write('FORMat:DATA {}'.format(strict_discrete_set(data_format, self.DATA_FORMAT)))
        self.write('FORMat:ELEMents {}'.format(','.join(elements)))
        self.write('FORMat:BORDer {}'.format('NORMal' if byte_order_normal else 'SWAPped'))

    def one_shot_measurement(self, mode=None):
        """
        Take a reading in :attr:'mode' (active mode if no :attr:'mode' provided) with default parameters. This command
        is equivalent to changing the mode, setting trigger source to immediate, setting trigger count to 1 and
        configuring measurement parameters to factory defaults.
        :param mode: Desired mode for measurement.
        :return: measurement string
        """
        return self.ask(':MEASure:{0}?'.format(self._mode_command(mode)))

    def multi_point_measurement(self, samples=1):
        """
        Takes 'samples' readings in the active mode. Clears buffer buffer before called.
        :param samples: samples to acquire (1 - 55000)
        :return: measurement string
        """
        self.write('TRACe:CLEar')
        self.write('SAMP:COUN {:d}'.format(truncated_range(samples, [1, 55000])))
        return self.ask(':READ?')
