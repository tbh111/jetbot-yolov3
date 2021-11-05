'''
@Copyright (C): 2010-2019, Shenzhen Yahboom Tech
@Author: Malloy.Yuan
@Date: 2019-07-17 10:10:02
@LastEditors: Malloy.Yuan
@LastEditTime: 2019-09-20 11:00:55
'''
import Adafruit_GPIO as GPIO

class Programing_RGB(object):
        
    def get_i2c_device(self,address, i2c, i2c_bus):
        if i2c is not None:
            return i2c.get_i2c_device(address)
        else:
            import Adafruit_GPIO.I2C as I2C
            if i2c_bus is None:
                return I2C.get_i2c_device(address)
            else:
                return I2C.get_i2c_device(address, busnum=i2c_bus)
        
    def __init__(self):
        # Create I2C device.
        #"""Initialize the RGB."""
        # Setup I2C interface for the device.
#         if i2c is None:
#             import Adafruit_GPIO.I2C as I2C
#             i2c = I2C
        self._device = self.get_i2c_device(0x1b, None, 1)
    
    def Set_All_RGB(self, R_Value, G_Value, B_Value):
        try:
            self._device.write8(0x00,0xFF)
            self._device.write8(0x01,R_Value)
            self._device.write8(0x02,G_Value)
            self._device.write8(0x03,B_Value)
        except:
            print ('Set_All_RGB I2C error')
    
    def OFF_ALL_RGB(self):
        try:
            self.Set_All_RGB(0x00,0x00,0x00)
        except:
            print ('OFF_ALL_RGB I2C error')
    
    def Set_An_RGB(self, Position, R_Value, G_Value, B_Value):
        try:
            if(Position <= 0x09):
                self._device.write8(0x00,Position)
                self._device.write8(0x01,R_Value)
                self._device.write8(0x02,G_Value)
                self._device.write8(0x03,B_Value)
        except:
            print ('Set_An_RGB I2C error')
    def Set_WaterfallLight_RGB(self):
        try:
            # self.OFF_ALL_RGB()
            self._device.write8(0x04, 0x00)
        except:
            print ('Set_WaterfallLight_RGB I2C error')
    def Set_BreathColor_RGB(self):
        try:
            # self.OFF_ALL_RGB()
            self._device.write8(0x04, 0x01)
        except:
            print ('Set_BreathColor_RGB I2C error')
    def Set_ChameleonLight_RGB(self):
        try:
            # self.OFF_ALL_RGB()
            self._device.write8(0x04, 0x02)
        except:
            print ('Set_ChameleonLight_RGB I2C error')
    #确保颜色值在0-6中
    def Set_BreathSColor_RGB(self, color):
        try:
            self._device.write8(0x05, color)
        except:
            print ('Set_BreathSColor_RGB I2C error')
    #确保速度设置值在1,2,3中
    def Set_BreathSSpeed_RGB(self, speed):
        try:
            self._device.write8(0x06, speed)
        except:
            print ('Set_BreathSSpeed_RGB I2C error')
    def Set_BreathSLight_RGB(self):
        try:
            # self.OFF_ALL_RGB()
            self._device.write8(0x04, 0x03)
        except:
            print ('Set_BreathSLight_RGB I2C error')