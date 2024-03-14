import unittest
import numpy
import torch
from nfpn import to_hf10, hf10_to_fp16, HF10_MAX

def is_subnormal(v):
    return v < 2 ** -14

def as_fp16(vi: int):
    vf = numpy.array([vi], dtype=numpy.uint16).view(numpy.float16)[0]
    return vf

class TestToHf12(unittest.TestCase):
    
    def test_normal_fp16_normal_hf10(self):
        xs = [
            # exp  value               fp16                   hf10 (expected)
            (-5,   0.03125,            0b0_01010_0000000000,  0b0_111_000000),
            (-6,   0.015625,           0b0_01001_0000000000,  0b0_110_000000),
            (-7,   0.0078125,          0b0_01000_0000000000,  0b0_101_000000),
            (-8,   0.00390625,         0b0_00111_0000000000,  0b0_100_000000),
            (-9,   0.001953125,        0b0_00110_0000000000,  0b0_011_000000),
            (-10,  0.0009765625,       0b0_00101_0000000000,  0b0_010_000000),
            (-11,  0.00048828125,      0b0_00100_0000000000,  0b0_001_000000),
            (-5,  -0.03125,            0b1_01010_0000000000,  0b1_111_000000),
            (-6,  -0.015625,           0b1_01001_0000000000,  0b1_110_000000),
            (-7,  -0.0078125,          0b1_01000_0000000000,  0b1_101_000000),
            (-8,  -0.00390625,         0b1_00111_0000000000,  0b1_100_000000),
            (-9,  -0.001953125,        0b1_00110_0000000000,  0b1_011_000000),
            (-10, -0.0009765625,       0b1_00101_0000000000,  0b1_010_000000),
            (-11, -0.00048828125,      0b1_00100_0000000000,  0b1_001_000000),
        ]
        
        for exp, x, fp16, hf10 in xs:
            xx = torch.tensor([x, x, x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[1].item(), x)
            self.assertEqual(xx[2].item(), x)
            self.assertEqual(xx[3].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_hf10(xx)
            
            self.assertEqual(e.shape, (1,))
            ev = e.item()
            self.assertEqual(ev >> 6, (ev >> 4) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 2) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 0) & 0b11)
            self.assertEqual(f.shape, (4,))
            self.assertEqual(f[0].item(), f[1].item())
            self.assertEqual(f[0].item(), f[2].item())
            self.assertEqual(f[0].item(), f[3].item())
            
            e = ev >> 6
            f = f[0].item()
            
            e_expected = hf10 >> 8
            f_expected = hf10 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
            

    def test_normal_fp16_subnormal_hf10(self):
        xs = [
            # exp  value               fp16                   hf10 (expected)
            (-1,   0.5,                0b0_01110_0000000000,  0b0_000_000_111),
            (-2,   0.25,               0b0_01101_0000000000,  0b0_000_000_110),
            (-3,   0.125,              0b0_01100_0000000000,  0b0_000_000_101),
            (-4,   0.0625,             0b0_01011_0000000000,  0b0_000_000_100),
            (-1,  -0.5,                0b1_01110_0000000000,  0b1_000_000_111),
            (-2,  -0.25,               0b1_01101_0000000000,  0b1_000_000_110),
            (-3,  -0.125,              0b1_01100_0000000000,  0b1_000_000_101),
            (-4,  -0.0625,             0b1_01011_0000000000,  0b1_000_000_100),
            (-12,  0.000244140625,     0b0_00011_0000000000,  0b0_000_000_011),
            (-13,  0.0001220703125,    0b0_00010_0000000000,  0b0_000_000_010),
            (-14,  6.103515625e-05,    0b0_00001_0000000000,  0b0_000_000_001),
            (-12, -0.000244140625,     0b1_00011_0000000000,  0b1_000_000_011),
            (-13, -0.0001220703125,    0b1_00010_0000000000,  0b1_000_000_010),
            (-14, -6.103515625e-05,    0b1_00001_0000000000,  0b1_000_000_001),
        ]
        
        for exp, x, fp16, hf10 in xs:
            xx = torch.tensor([x, x, x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[1].item(), x)
            self.assertEqual(xx[2].item(), x)
            self.assertEqual(xx[3].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_hf10(xx)
            
            self.assertEqual(e.shape, (1,))
            ev = e.item()
            self.assertEqual(ev >> 6, (ev >> 4) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 2) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 0) & 0b11)
            self.assertEqual(f.shape, (4,))
            self.assertEqual(f[0].item(), f[1].item())
            self.assertEqual(f[0].item(), f[2].item())
            self.assertEqual(f[0].item(), f[3].item())
            
            e = ev >> 6
            f = f[0].item()
            
            e_expected = hf10 >> 8
            f_expected = hf10 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
    
    def test_subnormal_fp16_subnormal_hf10(self):
        xs = [
            # exp  value                   fp16                  hf10 (expected)
            (-15, 3.0517578125e-05,        0b0_00000_1000000000, 0b0_000_100_000),
            (-16, 1.52587890625e-05,       0b0_00000_0100000000, 0b0_000_010_000),
            (-17, 7.62939453125e-06,       0b0_00000_0010000000, 0b0_000_001_000),
            (-18, 3.814697265625e-06,      0b0_00000_0001000000, 0b0_000_000_000),
            (-19, 1.9073486328125e-06,     0b0_00000_0000100000, 0b0_000_000_000),
            (-20, 9.5367431640625e-07,     0b0_00000_0000010000, 0b0_000_000_000),
            (-21, 4.76837158203125e-07,    0b0_00000_0000001000, 0b0_000_000_000),
            (-22, 2.384185791015625e-07,   0b0_00000_0000000100, 0b0_000_000_000),
            (-23, 1.1920928955078125e-07,  0b0_00000_0000000010, 0b0_000_000_000),
            (-24, 5.960464477539063e-08,   0b0_00000_0000000001, 0b0_000_000_000),
            (-15, -3.0517578125e-05,       0b1_00000_1000000000, 0b1_000_100_000),
            (-16, -1.52587890625e-05,      0b1_00000_0100000000, 0b1_000_010_000),
            (-17, -7.62939453125e-06,      0b1_00000_0010000000, 0b1_000_001_000),
            (-18, -3.814697265625e-06,     0b1_00000_0001000000, 0b1_000_000_000),
            (-19, -1.9073486328125e-06,    0b1_00000_0000100000, 0b1_000_000_000),
            (-20, -9.5367431640625e-07,    0b1_00000_0000010000, 0b1_000_000_000),
            (-21, -4.76837158203125e-07,   0b1_00000_0000001000, 0b1_000_000_000),
            (-22, -2.384185791015625e-07,  0b1_00000_0000000100, 0b1_000_000_000),
            (-23, -1.1920928955078125e-07, 0b1_00000_0000000010, 0b1_000_000_000),
            (-24, -5.960464477539063e-08,  0b1_00000_0000000001, 0b1_000_000_000),
        ]
        
        for exp, x, fp16, hf10 in xs:
            xx = torch.tensor([x, x, x, x], dtype=torch.float16)
            
            self.assertEqual(xx[0].item(), x)
            self.assertEqual(xx[1].item(), x)
            self.assertEqual(xx[2].item(), x)
            self.assertEqual(xx[3].item(), x)
            self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16)
            
            e, f = to_hf10(xx)
            
            self.assertEqual(e.shape, (1,))
            ev = e.item()
            self.assertEqual(ev >> 6, (ev >> 4) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 2) & 0b11)
            self.assertEqual(ev >> 6, (ev >> 0) & 0b11)
            self.assertEqual(f.shape, (4,))
            self.assertEqual(f[0].item(), f[1].item())
            self.assertEqual(f[0].item(), f[2].item())
            self.assertEqual(f[0].item(), f[3].item())
            
            e = ev >> 6
            f = f[0].item()
            
            e_expected = hf10 >> 8
            f_expected = hf10 & 0b1111_1111
            
            self.assertEqual(e, e_expected)
            self.assertEqual(f, f_expected)
    
    
    def test_normal_hf10_combine(self):
        xs = [
            # exp  value               fp16                   hf10 (expected)
            (-5,   0.03125,            0b0_01010_0000000000,  0b0_111_000000),
            (-6,   0.015625,           0b0_01001_0000000000,  0b0_110_000000),
            (-7,   0.0078125,          0b0_01000_0000000000,  0b0_101_000000),
            (-8,   0.00390625,         0b0_00111_0000000000,  0b0_100_000000),
            (-9,   0.001953125,        0b0_00110_0000000000,  0b0_011_000000),
            (-10,  0.0009765625,       0b0_00101_0000000000,  0b0_010_000000),
            (-11,  0.00048828125,      0b0_00100_0000000000,  0b0_001_000000),
            (-5,  -0.03125,            0b1_01010_0000000000,  0b1_111_000000),
            (-6,  -0.015625,           0b1_01001_0000000000,  0b1_110_000000),
            (-7,  -0.0078125,          0b1_01000_0000000000,  0b1_101_000000),
            (-8,  -0.00390625,         0b1_00111_0000000000,  0b1_100_000000),
            (-9,  -0.001953125,        0b1_00110_0000000000,  0b1_011_000000),
            (-10, -0.0009765625,       0b1_00101_0000000000,  0b1_010_000000),
            (-11, -0.00048828125,      0b1_00100_0000000000,  0b1_001_000000),
        ]
        
        fs = [
            # value        fp16            hf10 (expected)
            (1.5,          0b10_0000_0000, 0b10_0000),
            (1.25,         0b01_0000_0000, 0b01_0000),
            (1.125,        0b00_1000_0000, 0b00_1000),
            (1.0625,       0b00_0100_0000, 0b00_0100),
            (1.03125,      0b00_0010_0000, 0b00_0010),
            (1.015625,     0b00_0001_0000, 0b00_0001),
            (1.0078125,    0b00_0000_1000, 0b00_0000),
            (1.00390625,   0b00_0000_0100, 0b00_0000),
            (1.001953125,  0b00_0000_0010, 0b00_0000),
            (1.0009765625, 0b00_0000_0001, 0b00_0000),
            (1.0,          0b00_0000_0000, 0b00_0000),
        ]
        
        for exp, x0, fp16_e, hf10_e in xs:
            for x1, fp16_f, hf10_f in fs:
                x = x0 * x1
                
                xx = torch.tensor([x, x, x, x], dtype=torch.float16)
                
                self.assertEqual(xx[0].item(), x)
                self.assertEqual(xx[1].item(), x)
                self.assertEqual(xx[2].item(), x)
                self.assertEqual(xx[3].item(), x)
                self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16_e | fp16_f, [exp, x0, x1, fp16_e, fp16_f])
        
                e, f = to_hf10(xx)
                
                self.assertEqual(e.shape, (1,))
                ev = e.item()
                self.assertEqual(ev >> 6, (ev >> 4) & 0b11)
                self.assertEqual(ev >> 6, (ev >> 2) & 0b11)
                self.assertEqual(ev >> 6, (ev >> 0) & 0b11)
                self.assertEqual(f.shape, (4,))
                self.assertEqual(f[0].item(), f[1].item())
                self.assertEqual(f[0].item(), f[2].item())
                self.assertEqual(f[0].item(), f[3].item())
                
                e = ev >> 6
                f = f[0].item()
                
                e_expected = hf10_e >> 8
                f_expected = (hf10_e & 0b0_011_000000) | hf10_f
                
                self.assertEqual(e, e_expected, [exp, x0, x1, fp16_e, hf10_e, fp16_f, hf10_f])
                self.assertEqual(f, f_expected, [exp, x0, x1, fp16_e, hf10_e, fp16_f, hf10_f])
        

    def test_subnormal_hf10_combine(self):
        xs = [
            # exp  value               fp16                   hf10 (expected)
            (-1,   0.5,                0b0_01110_0000000000,  0b0_000_000_111),
            (-2,   0.25,               0b0_01101_0000000000,  0b0_000_000_110),
            (-3,   0.125,              0b0_01100_0000000000,  0b0_000_000_101),
            (-4,   0.0625,             0b0_01011_0000000000,  0b0_000_000_100),
            (-1,  -0.5,                0b1_01110_0000000000,  0b1_000_000_111),
            (-2,  -0.25,               0b1_01101_0000000000,  0b1_000_000_110),
            (-3,  -0.125,              0b1_01100_0000000000,  0b1_000_000_101),
            (-4,  -0.0625,             0b1_01011_0000000000,  0b1_000_000_100),
            (-12,  0.000244140625,     0b0_00011_0000000000,  0b0_000_000_011),
            (-13,  0.0001220703125,    0b0_00010_0000000000,  0b0_000_000_010),
            (-14,  6.103515625e-05,    0b0_00001_0000000000,  0b0_000_000_001),
            (-12, -0.000244140625,     0b1_00011_0000000000,  0b1_000_000_011),
            (-13, -0.0001220703125,    0b1_00010_0000000000,  0b1_000_000_010),
            (-14, -6.103515625e-05,    0b1_00001_0000000000,  0b1_000_000_001),
        ]
        
        fs = [
            # value        fp16            hf10 (expected)
            (1.5,          0b10_0000_0000, 0b10_0000),
            (1.25,         0b01_0000_0000, 0b01_0000),
            (1.125,        0b00_1000_0000, 0b00_1000),
            (1.0625,       0b00_0100_0000, 0b00_0100),
            (1.03125,      0b00_0010_0000, 0b00_0010),
            (1.015625,     0b00_0001_0000, 0b00_0001),
            (1.0078125,    0b00_0000_1000, 0b00_0000),
            (1.00390625,   0b00_0000_0100, 0b00_0000),
            (1.001953125,  0b00_0000_0010, 0b00_0000),
            (1.0009765625, 0b00_0000_0001, 0b00_0000),
            (1.0,          0b00_0000_0000, 0b00_0000),
        ]
        
        for exp, x0, fp16_e, hf10_e in xs:
            for x1, fp16_f, hf10_f in fs:
                x = x0 * x1
                
                xx = torch.tensor([x, x, x, x], dtype=torch.float16)
                
                self.assertEqual(xx[0].item(), x)
                self.assertEqual(xx[1].item(), x)
                self.assertEqual(xx[2].item(), x)
                self.assertEqual(xx[3].item(), x)
                self.assertEqual(xx[0].view(dtype=torch.int16).item() & 0xffff, fp16_e | fp16_f)
        
                e, f = to_hf10(xx)
                
                self.assertEqual(e.shape, (1,))
                ev = e.item()
                self.assertEqual(ev >> 6, (ev >> 4) & 0b11)
                self.assertEqual(ev >> 6, (ev >> 2) & 0b11)
                self.assertEqual(ev >> 6, (ev >> 0) & 0b11)
                self.assertEqual(f.shape, (4,))
                self.assertEqual(f[0].item(), f[1].item())
                self.assertEqual(f[0].item(), f[2].item())
                self.assertEqual(f[0].item(), f[3].item())
                
                e = ev >> 6
                f = f[0].item()
                
                e_expected = hf10_e >> 8
                f_expected = (hf10_f & 0b111000) | (hf10_e & 0b0000_0111)
                
                self.assertEqual(e, e_expected)
                self.assertEqual(f, f_expected, [exp, x0, x1, f'{f:08b}', f'{f_expected:08b}'])

    
class TestToFp16(unittest.TestCase):
    
    def test_normal_hf10(self):
        es = [
            # exp hf10     fp16
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
        
        for exp, hf10_e, fp16_e in es:
            hf10_e1 = hf10_e >> 2
            hf10_e2 = hf10_e & 0b11
            ee = torch.tensor([(hf10_e1 << 6) | (hf10_e1 << 4) | (hf10_e1 << 2) | hf10_e1], dtype=torch.uint8)
            ff = torch.tensor([hf10_e2 << 6], dtype=torch.uint8).repeat(4)
            
            xs = hf10_to_fp16(ee, ff)
            
            self.assertEqual(xs.shape, (4,))
            self.assertEqual(xs[0].item(), xs[1].item())
            self.assertEqual(xs[0].item(), xs[2].item())
            self.assertEqual(xs[0].item(), xs[3].item())
            
            x = xs[0].view(dtype=torch.int16).item() & 0xffff
            
            self.assertEqual(x & 0b11_1111_1111, 0)
            self.assertEqual(x >> 10, fp16_e, [exp, hf10_e, f'{ee.item():08b}', ff, x, fp16_e])
        
        for exp, hf10_e, fp16_e in es:
            for hf10_f in range(1<<6):
                fp16_f = hf10_f << 4
        
                hf10_e1 = hf10_e >> 2
                hf10_e2 = hf10_e & 0b11
                ee = torch.tensor([(hf10_e1 << 6) | (hf10_e1 << 4) | (hf10_e1 << 2) | hf10_e1], dtype=torch.uint8)
                ff = torch.tensor([(hf10_e2 << 6) | hf10_f], dtype=torch.uint8).repeat(4)
        
                xs = hf10_to_fp16(ee, ff)
                
                self.assertEqual(xs.shape, (4,))
                self.assertEqual(xs[0].item(), xs[1].item())
                self.assertEqual(xs[0].item(), xs[2].item())
                self.assertEqual(xs[0].item(), xs[3].item())
                
                x = xs[0].view(dtype=torch.int16).item() & 0xffff
                
                self.assertEqual(x & 0b11_1111_1111, fp16_f, f'{exp} {hf10_e:04b}:{hf10_f:08b} {fp16_e:06b}:{fp16_f:010b} {x&0b11_1111_1111:010b}')
                self.assertEqual(x >> 10, fp16_e)
            
            
    def test_subnormal_hf10_normal_fp16(self):
        pass
    
    def test_subnormal_hf10_subnormal_fp16(self):
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
        
        if HF10_MAX <= as_fp16(x):
            continue
        
        xx = torch.tensor([x, x, x, x], dtype=torch.int16).view(dtype=torch.float16)
        if s == 1:
            xx = -xx
            x = (s << 15) | x
        
        ee, ff = to_hf10(xx)
        
        yy = hf10_to_fp16(ee, ff)
        
        assert yy.shape == (4,)
        assert yy[0] == yy[1]
        assert yy[0] == yy[2]
        assert yy[0] == yy[3]
        
        y = yy[0].view(dtype=torch.int16).item() & 0xffff
        
        xf, yf = numpy.array([x, y], dtype=numpy.uint16).view(numpy.float16)
        if e in range(-11, -4):
            d = 1 << 4
        else:
            d = 1 << 7
        assert abs(x - y) < d, f'{yf-xf}   x={xf} ({x} {x:016b}), y={yf} ({y} {y:016b}), s={s}, e={e-15} ({e:05b}), f={f:010b}, ee={ee.item() >> 4:04b}, ff={ff[0].item():08b}'
        
        i += 1


if __name__ == '__main__':
    unittest.main()
    #test_rand(1024 * 10)
