import unittest
import numpy
import torch
from fp12 import to_fp12, fp12_to_fp16, FP12_MAX

def is_subnormal(v):
    return v < 2 ** -14

def as_fp16(vi: int):
    vf = numpy.array([vi], dtype=numpy.uint16).view(numpy.float16)[0]
    return vf

class TestToFp12(unittest.TestCase):
    
    def test_normal_fp16_normal_fp12(self):
        xs = [
            # exp  value               fp16                   fp12 (expected)
            (-5,   0.03125,            0b0_01010_0000000000,  0b0_111_00000000),
            (-6,   0.015625,           0b0_01001_0000000000,  0b0_110_00000000),
            (-7,   0.0078125,          0b0_01000_0000000000,  0b0_101_00000000),
            (-8,   0.00390625,         0b0_00111_0000000000,  0b0_100_00000000),
            (-9,   0.001953125,        0b0_00110_0000000000,  0b0_011_00000000),
            (-10,  0.0009765625,       0b0_00101_0000000000,  0b0_010_00000000),
            (-11,  0.00048828125,      0b0_00100_0000000000,  0b0_001_00000000),
            (-5,  -0.03125,            0b1_01010_0000000000,  0b1_111_00000000),
            (-6,  -0.015625,           0b1_01001_0000000000,  0b1_110_00000000),
            (-7,  -0.0078125,          0b1_01000_0000000000,  0b1_101_00000000),
            (-8,  -0.00390625,         0b1_00111_0000000000,  0b1_100_00000000),
            (-9,  -0.001953125,        0b1_00110_0000000000,  0b1_011_00000000),
            (-10, -0.0009765625,       0b1_00101_0000000000,  0b1_010_00000000),
            (-11, -0.00048828125,      0b1_00100_0000000000,  0b1_001_00000000),
        ]
        
        for exp, x, fp16, fp12 in xs:
            xx = torch.tensor([x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_fp12(xx)
            #y = fp12_to_fp16(e, f)
            
            self.assertEqual(e.shape, (1,))
            self.assertEqual(e.item() >> 4, e.item() & 0b1111)
            self.assertEqual(f.shape, (2,))
            self.assertEqual(f[0].item(), f[1].item())
            
            e = e.item() >> 4
            f = f[0].item()
            
            e_expected = fp12 >> 8
            f_expected = fp12 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
            

    def test_normal_fp16_subnormal_fp12(self):
        xs = [
            # exp  value               fp16                   fp12 (expected)
            (-1,   0.5,                0b0_01110_0000000000,  0b0_000_00000_111),
            (-2,   0.25,               0b0_01101_0000000000,  0b0_000_00000_110),
            (-3,   0.125,              0b0_01100_0000000000,  0b0_000_00000_101),
            (-4,   0.0625,             0b0_01011_0000000000,  0b0_000_00000_100),
            (-1,  -0.5,                0b1_01110_0000000000,  0b1_000_00000_111),
            (-2,  -0.25,               0b1_01101_0000000000,  0b1_000_00000_110),
            (-3,  -0.125,              0b1_01100_0000000000,  0b1_000_00000_101),
            (-4,  -0.0625,             0b1_01011_0000000000,  0b1_000_00000_100),
            (-12,  0.000244140625,     0b0_00011_0000000000,  0b0_000_00000_011),
            (-13,  0.0001220703125,    0b0_00010_0000000000,  0b0_000_00000_010),
            (-14,  6.103515625e-05,    0b0_00001_0000000000,  0b0_000_00000_001),
            (-12, -0.000244140625,     0b1_00011_0000000000,  0b1_000_00000_011),
            (-13, -0.0001220703125,    0b1_00010_0000000000,  0b1_000_00000_010),
            (-14, -6.103515625e-05,    0b1_00001_0000000000,  0b1_000_00000_001),
        ]
        
        for exp, x, fp16, fp12 in xs:
            xx = torch.tensor([x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_fp12(xx)
            #y = fp12_to_fp16(e, f)
            
            self.assertEqual(e.shape, (1,))
            self.assertEqual(e.item() >> 4, e.item() & 0b1111)
            self.assertEqual(f.shape, (2,))
            self.assertEqual(f[0].item(), f[1].item())
            
            e = e.item() >> 4
            f = f[0].item()
            
            e_expected = fp12 >> 8
            f_expected = fp12 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
    
    def test_subnormal_fp16_subnormal_fp12(self):
        xs = [
            # exp  value                   fp16                  fp12 (expected)
            (-15, 3.0517578125e-05,        0b0_00000_1000000000, 0b0_000_10000_000),
            (-16, 1.52587890625e-05,       0b0_00000_0100000000, 0b0_000_01000_000),
            (-17, 7.62939453125e-06,       0b0_00000_0010000000, 0b0_000_00100_000),
            (-18, 3.814697265625e-06,      0b0_00000_0001000000, 0b0_000_00010_000),
            (-19, 1.9073486328125e-06,     0b0_00000_0000100000, 0b0_000_00001_000),
            (-20, 9.5367431640625e-07,     0b0_00000_0000010000, 0b0_000_00000_000),
            (-21, 4.76837158203125e-07,    0b0_00000_0000001000, 0b0_000_00000_000),
            (-22, 2.384185791015625e-07,   0b0_00000_0000000100, 0b0_000_00000_000),
            (-23, 1.1920928955078125e-07,  0b0_00000_0000000010, 0b0_000_00000_000),
            (-24, 5.960464477539063e-08,   0b0_00000_0000000001, 0b0_000_00000_000),
            (-15, -3.0517578125e-05,       0b1_00000_1000000000, 0b1_000_10000_000),
            (-16, -1.52587890625e-05,      0b1_00000_0100000000, 0b1_000_01000_000),
            (-17, -7.62939453125e-06,      0b1_00000_0010000000, 0b1_000_00100_000),
            (-18, -3.814697265625e-06,     0b1_00000_0001000000, 0b1_000_00010_000),
            (-19, -1.9073486328125e-06,    0b1_00000_0000100000, 0b1_000_00001_000),
            (-20, -9.5367431640625e-07,    0b1_00000_0000010000, 0b1_000_00000_000),
            (-21, -4.76837158203125e-07,   0b1_00000_0000001000, 0b1_000_00000_000),
            (-22, -2.384185791015625e-07,  0b1_00000_0000000100, 0b1_000_00000_000),
            (-23, -1.1920928955078125e-07, 0b1_00000_0000000010, 0b1_000_00000_000),
            (-24, -5.960464477539063e-08,  0b1_00000_0000000001, 0b1_000_00000_000),
        ]
        
        for exp, x, fp16, fp12 in xs:
            xx = torch.tensor([x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_fp12(xx)
            #y = fp12_to_fp16(e, f)
            
            self.assertEqual(e.shape, (1,))
            self.assertEqual(e.item() >> 4, e.item() & 0b1111)
            self.assertEqual(f.shape, (2,))
            self.assertEqual(f[0].item(), f[1].item())
            
            e = e.item() >> 4
            f = f[0].item()
            
            e_expected = fp12 >> 8
            f_expected = fp12 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
    
    
    def test_normal_fp12_combine(self):
        xs = [
            # exp  value               fp16                   fp12 (expected)
            (-5,   0.03125,            0b0_01010_0000000000,  0b0_111_00000000),
            (-6,   0.015625,           0b0_01001_0000000000,  0b0_110_00000000),
            (-7,   0.0078125,          0b0_01000_0000000000,  0b0_101_00000000),
            (-8,   0.00390625,         0b0_00111_0000000000,  0b0_100_00000000),
            (-9,   0.001953125,        0b0_00110_0000000000,  0b0_011_00000000),
            (-10,  0.0009765625,       0b0_00101_0000000000,  0b0_010_00000000),
            (-11,  0.00048828125,      0b0_00100_0000000000,  0b0_001_00000000),
            (-5,  -0.03125,            0b1_01010_0000000000,  0b1_111_00000000),
            (-6,  -0.015625,           0b1_01001_0000000000,  0b1_110_00000000),
            (-7,  -0.0078125,          0b1_01000_0000000000,  0b1_101_00000000),
            (-8,  -0.00390625,         0b1_00111_0000000000,  0b1_100_00000000),
            (-9,  -0.001953125,        0b1_00110_0000000000,  0b1_011_00000000),
            (-10, -0.0009765625,       0b1_00101_0000000000,  0b1_010_00000000),
            (-11, -0.00048828125,      0b1_00100_0000000000,  0b1_001_00000000),
        ]
        
        fs = [
            # value        fp16            fp12 (expected)
            (1.5,          0b10_0000_0000, 0b1000_0000),
            (1.25,         0b01_0000_0000, 0b0100_0000),
            (1.125,        0b00_1000_0000, 0b0010_0000),
            (1.0625,       0b00_0100_0000, 0b0001_0000),
            (1.03125,      0b00_0010_0000, 0b0000_1000),
            (1.015625,     0b00_0001_0000, 0b0000_0100),
            (1.0078125,    0b00_0000_1000, 0b0000_0010),
            (1.00390625,   0b00_0000_0100, 0b0000_0001),
            (1.001953125,  0b00_0000_0010, 0b0000_0000),
            (1.0009765625, 0b00_0000_0001, 0b0000_0000),
            (1.0,          0b00_0000_0000, 0b0000_0000),
        ]
        
        for exp, x0, fp16_e, fp12_e in xs:
            for x1, fp16_f, fp12_f in fs:
                x = x0 * x1
                
                xx = torch.tensor([x, x], dtype=torch.float16)
                
                self.assertEqual(xx[0].item(), x)
                self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16_e | fp16_f, [exp, x0, x1, fp16_e, fp16_f])
        
                e, f = to_fp12(xx)
                
                self.assertEqual(e.shape, (1,))
                self.assertEqual(e.item() >> 4, e.item() & 0b1111)
                self.assertEqual(f.shape, (2,))
                self.assertEqual(f[0].item(), f[1].item())
                
                e = e.item() >> 4
                f = f[0].item()
                
                e_expected = fp12_e >> 8
                f_expected = fp12_f
                
                self.assertEqual(e, e_expected, [exp, x0, x1, fp16_e, fp12_e, fp16_f, fp12_f])
                self.assertEqual(f, f_expected, [exp, x0, x1, fp16_e, fp12_e, fp16_f, fp12_f])
        

    def test_subnormal_fp12_combine(self):
        xs = [
            # exp  value               fp16                   fp12 (expected)
            (-1,   0.5,                0b0_01110_0000000000,  0b0_000_00000_111),
            (-2,   0.25,               0b0_01101_0000000000,  0b0_000_00000_110),
            (-3,   0.125,              0b0_01100_0000000000,  0b0_000_00000_101),
            (-4,   0.0625,             0b0_01011_0000000000,  0b0_000_00000_100),
            (-1,  -0.5,                0b1_01110_0000000000,  0b1_000_00000_111),
            (-2,  -0.25,               0b1_01101_0000000000,  0b1_000_00000_110),
            (-3,  -0.125,              0b1_01100_0000000000,  0b1_000_00000_101),
            (-4,  -0.0625,             0b1_01011_0000000000,  0b1_000_00000_100),
            (-12,  0.000244140625,     0b0_00011_0000000000,  0b0_000_00000_011),
            (-13,  0.0001220703125,    0b0_00010_0000000000,  0b0_000_00000_010),
            (-14,  6.103515625e-05,    0b0_00001_0000000000,  0b0_000_00000_001),
            (-12, -0.000244140625,     0b1_00011_0000000000,  0b1_000_00000_011),
            (-13, -0.0001220703125,    0b1_00010_0000000000,  0b1_000_00000_010),
            (-14, -6.103515625e-05,    0b1_00001_0000000000,  0b1_000_00000_001),
        ]
        
        fs = [
            # value        fp16            fp12 (expected)
            (1.5,          0b10_0000_0000, 0b1000_0000),
            (1.25,         0b01_0000_0000, 0b0100_0000),
            (1.125,        0b00_1000_0000, 0b0010_0000),
            (1.0625,       0b00_0100_0000, 0b0001_0000),
            (1.03125,      0b00_0010_0000, 0b0000_1000),
            (1.015625,     0b00_0001_0000, 0b0000_0100),
            (1.0078125,    0b00_0000_1000, 0b0000_0010),
            (1.00390625,   0b00_0000_0100, 0b0000_0001),
            (1.001953125,  0b00_0000_0010, 0b0000_0000),
            (1.0009765625, 0b00_0000_0001, 0b0000_0000),
            (1.0,          0b00_0000_0000, 0b0000_0000),
        ]
        
        for exp, x0, fp16_e, fp12_e in xs:
            for x1, fp16_f, fp12_f in fs:
                x = x0 * x1
                
                xx = torch.tensor([x, x], dtype=torch.float16)
                
                self.assertEqual(xx[0].item(), x)
                self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16_e | fp16_f)
        
                e, f = to_fp12(xx)
                
                self.assertEqual(e.shape, (1,))
                self.assertEqual(e.item() >> 4, e.item() & 0b1111)
                self.assertEqual(f.shape, (2,))
                self.assertEqual(f[0].item(), f[1].item())
                
                e = e.item() >> 4
                f = f[0].item()
                
                e_expected = fp12_e >> 8
                f_expected = (fp12_f & 0b1111_1000) | (fp12_e & 0b0000_0111)
                
                self.assertEqual(e, e_expected)
                self.assertEqual(f, f_expected, [exp, x0, x1, f'{f:08b}', f'{f_expected:08b}'])

    
class TestToFp16(unittest.TestCase):
    
    def test_normal_fp12(self):
        es = [
            # exp fp12     fp16
            (-11, 0b0_001, 0b0_00100),
            (-10, 0b0_010, 0b0_00101),
            (-9,  0b0_011, 0b0_00110),
            (-8,  0b0_100, 0b0_00111),
            (-7,  0b0_101, 0b0_01000),
            (-6,  0b0_110, 0b0_01001),
            (-5,  0b0_111, 0b0_01010),
            (-11, 0b1_001, 0b1_00100),
            (-10, 0b1_010, 0b1_00101),
            (-9,  0b1_011, 0b1_00110),
            (-8,  0b1_100, 0b1_00111),
            (-7,  0b1_101, 0b1_01000),
            (-6,  0b1_110, 0b1_01001),
            (-5,  0b1_111, 0b1_01010),
        ]
        
        for exp, fp12_e, fp16_e in es:
            ee = torch.tensor([(fp12_e << 4) | fp12_e], dtype=torch.uint8)
            ff = torch.tensor([0, 0], dtype=torch.uint8)
            
            xs = fp12_to_fp16(ee, ff)
            
            self.assertEqual(xs.shape, (2,))
            self.assertEqual(xs[0].item(), xs[1].item())
            
            x = xs[0].view(dtype=torch.int16).item() & 0xffff
            
            self.assertEqual(x & 0b11_1111_1111, 0)
            self.assertEqual(x >> 10, fp16_e, [exp, fp12_e, f'{ee.item():08b}', ff, x, fp16_e])
        
        for exp, fp12_e, fp16_e in es:
            for fp12_f in range(0x100):
                fp16_f = fp12_f << 2
        
                ee = torch.tensor([(fp12_e << 4) | fp12_e], dtype=torch.uint8)
                ff = torch.tensor([fp12_f, fp12_f], dtype=torch.uint8)
        
                xs = fp12_to_fp16(ee, ff)
                
                self.assertEqual(xs.shape, (2,))
                self.assertEqual(xs[0].item(), xs[1].item())
                
                x = xs[0].view(dtype=torch.int16).item() & 0xffff
                
                self.assertEqual(x & 0b11_1111_1111, fp16_f, f'{exp} {fp12_e:04b}:{fp12_f:08b} {fp16_e:06b}:{fp16_f:010b} {x&0b11_1111_1111:010b}')
                self.assertEqual(x >> 10, fp16_e)
            
            
    def test_subnormal_fp12_normal_fp16(self):
        pass
    
    def test_subnormal_fp12_subnormal_fp16(self):
        pass


def test_rand(n=1024, seed=-1):
    import random
    
    if 0 <= seed:
        random.seed(seed)
    
    i = 0
    while i < n:
        s = random.randint(0, 1)
        e = random.randint(-11, -1) + 15
        f = random.randint(0, 0b0000_0011_1111_1111)
        
        x = (e << 10) | f
        
        if FP12_MAX <= as_fp16(x):
            continue
        
        xx = torch.tensor([x, x], dtype=torch.int16).view(dtype=torch.float16)
        if s == 1:
            xx = -xx
            x = (s << 15) | x
        
        ee, ff = to_fp12(xx)
        
        yy = fp12_to_fp16(ee, ff)
        
        assert yy.shape == (2,)
        assert yy[0] == yy[1]
        
        y = yy[0].view(dtype=torch.int16).item() & 0xffff
        
        xf, yf = numpy.array([x, y], dtype=numpy.uint16).view(numpy.float16)
        if e in range(-11, -4):
            d = 1 << 2
        else:
            d = 1 << 5
        assert abs(x - y) < d, f'{yf-xf}   x={xf} ({x} {x:016b}), y={yf} ({y} {y:016b}), s={s}, e={e-15} ({e:05b}), f={f:010b}, ee={ee.item() >> 4:04b}, ff={ff[0].item():08b}'
        
        i += 1


if __name__ == '__main__':
    unittest.main()
    #test_rand(1024 * 10)
