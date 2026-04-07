module @jit__forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["X"=1]>
  func.func public @main(%arg0: tensor<1024x16xf32>, %arg1: tensor<16x2048xf32>, %arg2: tensor<1024x16xf32>, %arg3: tensor<16x1024xf32>, %arg4: tensor<1024x16xf32>, %arg5: tensor<16x2048xf32>, %arg6: tensor<1024x16xf32>, %arg7: tensor<16x1024xf32>, %arg8: tensor<1024x16xf32>, %arg9: tensor<16x2048xf32>, %arg10: tensor<1024x16xf32>, %arg11: tensor<16x1024xf32>, %arg12: tensor<1024x16xf32>, %arg13: tensor<16x2048xf32>, %arg14: tensor<1024x16xf32>, %arg15: tensor<16x1024xf32>, %arg16: tensor<1024x16xf32>, %arg17: tensor<16x2048xf32>, %arg18: tensor<1024x16xf32>, %arg19: tensor<16x1024xf32>, %arg20: tensor<1024x16xf32>, %arg21: tensor<16x2048xf32>, %arg22: tensor<1024x16xf32>, %arg23: tensor<16x1024xf32>, %arg24: tensor<1024x16xf32>, %arg25: tensor<16x2048xf32>, %arg26: tensor<1024x16xf32>, %arg27: tensor<16x1024xf32>, %arg28: tensor<1024x16xf32>, %arg29: tensor<16x2048xf32>, %arg30: tensor<1024x16xf32>, %arg31: tensor<16x1024xf32>, %arg32: tensor<1024x16xf32>, %arg33: tensor<16x2048xf32>, %arg34: tensor<1024x16xf32>, %arg35: tensor<16x1024xf32>, %arg36: tensor<1024x16xf32>, %arg37: tensor<16x2048xf32>, %arg38: tensor<1024x16xf32>, %arg39: tensor<16x1024xf32>, %arg40: tensor<1024x16xf32>, %arg41: tensor<16x2048xf32>, %arg42: tensor<1024x16xf32>, %arg43: tensor<16x1024xf32>, %arg44: tensor<1024x16xf32>, %arg45: tensor<16x2048xf32>, %arg46: tensor<1024x16xf32>, %arg47: tensor<16x1024xf32>, %arg48: tensor<1024x16xf32>, %arg49: tensor<16x2048xf32>, %arg50: tensor<1024x16xf32>, %arg51: tensor<16x1024xf32>, %arg52: tensor<1024x16xf32>, %arg53: tensor<16x2048xf32>, %arg54: tensor<1024x16xf32>, %arg55: tensor<16x1024xf32>, %arg56: tensor<1024x16xf32>, %arg57: tensor<16x2048xf32>, %arg58: tensor<1024x16xf32>, %arg59: tensor<16x1024xf32>, %arg60: tensor<1024x16xf32>, %arg61: tensor<16x2048xf32>, %arg62: tensor<1024x16xf32>, %arg63: tensor<16x1024xf32>, %arg64: tensor<1024x16xf32>, %arg65: tensor<16x2048xf32>, %arg66: tensor<1024x16xf32>, %arg67: tensor<16x1024xf32>, %arg68: tensor<1024x16xf32>, %arg69: tensor<16x2048xf32>, %arg70: tensor<1024x16xf32>, %arg71: tensor<16x1024xf32>, %arg72: tensor<1024x16xf32>, %arg73: tensor<16x2048xf32>, %arg74: tensor<1024x16xf32>, %arg75: tensor<16x1024xf32>, %arg76: tensor<1024x16xf32>, %arg77: tensor<16x2048xf32>, %arg78: tensor<1024x16xf32>, %arg79: tensor<16x1024xf32>, %arg80: tensor<1024x16xf32>, %arg81: tensor<16x2048xf32>, %arg82: tensor<1024x16xf32>, %arg83: tensor<16x1024xf32>, %arg84: tensor<1024x16xf32>, %arg85: tensor<16x2048xf32>, %arg86: tensor<1024x16xf32>, %arg87: tensor<16x1024xf32>, %arg88: tensor<1024x16xf32>, %arg89: tensor<16x2048xf32>, %arg90: tensor<1024x16xf32>, %arg91: tensor<16x1024xf32>, %arg92: tensor<1024x16xf32>, %arg93: tensor<16x2048xf32>, %arg94: tensor<1024x16xf32>, %arg95: tensor<16x1024xf32>, %arg96: tensor<1024x16xf32>, %arg97: tensor<16x2048xf32>, %arg98: tensor<1024x16xf32>, %arg99: tensor<16x1024xf32>, %arg100: tensor<1024x16xf32>, %arg101: tensor<16x2048xf32>, %arg102: tensor<1024x16xf32>, %arg103: tensor<16x1024xf32>, %arg104: tensor<1024x16xf32>, %arg105: tensor<16x2048xf32>, %arg106: tensor<1024x16xf32>, %arg107: tensor<16x1024xf32>, %arg108: tensor<1024x16xf32>, %arg109: tensor<16x2048xf32>, %arg110: tensor<1024x16xf32>, %arg111: tensor<16x1024xf32>, %arg112: tensor<151936x1024xf32>, %arg113: tensor<1024xf32>, %arg114: tensor<3072x1024xf32>, %arg115: tensor<1024x3072xf32>, %arg116: tensor<1024x3072xf32>, %arg117: tensor<1024xf32>, %arg118: tensor<128xf32>, %arg119: tensor<1024x1024xf32>, %arg120: tensor<2048x1024xf32>, %arg121: tensor<128xf32>, %arg122: tensor<1024x2048xf32>, %arg123: tensor<1024x1024xf32>, %arg124: tensor<1024xf32>, %arg125: tensor<3072x1024xf32>, %arg126: tensor<1024x3072xf32>, %arg127: tensor<1024x3072xf32>, %arg128: tensor<1024xf32>, %arg129: tensor<128xf32>, %arg130: tensor<1024x1024xf32>, %arg131: tensor<2048x1024xf32>, %arg132: tensor<128xf32>, %arg133: tensor<1024x2048xf32>, %arg134: tensor<1024x1024xf32>, %arg135: tensor<1024xf32>, %arg136: tensor<3072x1024xf32>, %arg137: tensor<1024x3072xf32>, %arg138: tensor<1024x3072xf32>, %arg139: tensor<1024xf32>, %arg140: tensor<128xf32>, %arg141: tensor<1024x1024xf32>, %arg142: tensor<2048x1024xf32>, %arg143: tensor<128xf32>, %arg144: tensor<1024x2048xf32>, %arg145: tensor<1024x1024xf32>, %arg146: tensor<1024xf32>, %arg147: tensor<3072x1024xf32>, %arg148: tensor<1024x3072xf32>, %arg149: tensor<1024x3072xf32>, %arg150: tensor<1024xf32>, %arg151: tensor<128xf32>, %arg152: tensor<1024x1024xf32>, %arg153: tensor<2048x1024xf32>, %arg154: tensor<128xf32>, %arg155: tensor<1024x2048xf32>, %arg156: tensor<1024x1024xf32>, %arg157: tensor<1024xf32>, %arg158: tensor<3072x1024xf32>, %arg159: tensor<1024x3072xf32>, %arg160: tensor<1024x3072xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<128xf32>, %arg163: tensor<1024x1024xf32>, %arg164: tensor<2048x1024xf32>, %arg165: tensor<128xf32>, %arg166: tensor<1024x2048xf32>, %arg167: tensor<1024x1024xf32>, %arg168: tensor<1024xf32>, %arg169: tensor<3072x1024xf32>, %arg170: tensor<1024x3072xf32>, %arg171: tensor<1024x3072xf32>, %arg172: tensor<1024xf32>, %arg173: tensor<128xf32>, %arg174: tensor<1024x1024xf32>, %arg175: tensor<2048x1024xf32>, %arg176: tensor<128xf32>, %arg177: tensor<1024x2048xf32>, %arg178: tensor<1024x1024xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<3072x1024xf32>, %arg181: tensor<1024x3072xf32>, %arg182: tensor<1024x3072xf32>, %arg183: tensor<1024xf32>, %arg184: tensor<128xf32>, %arg185: tensor<1024x1024xf32>, %arg186: tensor<2048x1024xf32>, %arg187: tensor<128xf32>, %arg188: tensor<1024x2048xf32>, %arg189: tensor<1024x1024xf32>, %arg190: tensor<1024xf32>, %arg191: tensor<3072x1024xf32>, %arg192: tensor<1024x3072xf32>, %arg193: tensor<1024x3072xf32>, %arg194: tensor<1024xf32>, %arg195: tensor<128xf32>, %arg196: tensor<1024x1024xf32>, %arg197: tensor<2048x1024xf32>, %arg198: tensor<128xf32>, %arg199: tensor<1024x2048xf32>, %arg200: tensor<1024x1024xf32>, %arg201: tensor<1024xf32>, %arg202: tensor<3072x1024xf32>, %arg203: tensor<1024x3072xf32>, %arg204: tensor<1024x3072xf32>, %arg205: tensor<1024xf32>, %arg206: tensor<128xf32>, %arg207: tensor<1024x1024xf32>, %arg208: tensor<2048x1024xf32>, %arg209: tensor<128xf32>, %arg210: tensor<1024x2048xf32>, %arg211: tensor<1024x1024xf32>, %arg212: tensor<1024xf32>, %arg213: tensor<3072x1024xf32>, %arg214: tensor<1024x3072xf32>, %arg215: tensor<1024x3072xf32>, %arg216: tensor<1024xf32>, %arg217: tensor<128xf32>, %arg218: tensor<1024x1024xf32>, %arg219: tensor<2048x1024xf32>, %arg220: tensor<128xf32>, %arg221: tensor<1024x2048xf32>, %arg222: tensor<1024x1024xf32>, %arg223: tensor<1024xf32>, %arg224: tensor<3072x1024xf32>, %arg225: tensor<1024x3072xf32>, %arg226: tensor<1024x3072xf32>, %arg227: tensor<1024xf32>, %arg228: tensor<128xf32>, %arg229: tensor<1024x1024xf32>, %arg230: tensor<2048x1024xf32>, %arg231: tensor<128xf32>, %arg232: tensor<1024x2048xf32>, %arg233: tensor<1024x1024xf32>, %arg234: tensor<1024xf32>, %arg235: tensor<3072x1024xf32>, %arg236: tensor<1024x3072xf32>, %arg237: tensor<1024x3072xf32>, %arg238: tensor<1024xf32>, %arg239: tensor<128xf32>, %arg240: tensor<1024x1024xf32>, %arg241: tensor<2048x1024xf32>, %arg242: tensor<128xf32>, %arg243: tensor<1024x2048xf32>, %arg244: tensor<1024x1024xf32>, %arg245: tensor<1024xf32>, %arg246: tensor<3072x1024xf32>, %arg247: tensor<1024x3072xf32>, %arg248: tensor<1024x3072xf32>, %arg249: tensor<1024xf32>, %arg250: tensor<128xf32>, %arg251: tensor<1024x1024xf32>, %arg252: tensor<2048x1024xf32>, %arg253: tensor<128xf32>, %arg254: tensor<1024x2048xf32>, %arg255: tensor<1024x1024xf32>, %arg256: tensor<1024xf32>, %arg257: tensor<3072x1024xf32>, %arg258: tensor<1024x3072xf32>, %arg259: tensor<1024x3072xf32>, %arg260: tensor<1024xf32>, %arg261: tensor<128xf32>, %arg262: tensor<1024x1024xf32>, %arg263: tensor<2048x1024xf32>, %arg264: tensor<128xf32>, %arg265: tensor<1024x2048xf32>, %arg266: tensor<1024x1024xf32>, %arg267: tensor<1024xf32>, %arg268: tensor<3072x1024xf32>, %arg269: tensor<1024x3072xf32>, %arg270: tensor<1024x3072xf32>, %arg271: tensor<1024xf32>, %arg272: tensor<128xf32>, %arg273: tensor<1024x1024xf32>, %arg274: tensor<2048x1024xf32>, %arg275: tensor<128xf32>, %arg276: tensor<1024x2048xf32>, %arg277: tensor<1024x1024xf32>, %arg278: tensor<1024xf32>, %arg279: tensor<3072x1024xf32>, %arg280: tensor<1024x3072xf32>, %arg281: tensor<1024x3072xf32>, %arg282: tensor<1024xf32>, %arg283: tensor<128xf32>, %arg284: tensor<1024x1024xf32>, %arg285: tensor<2048x1024xf32>, %arg286: tensor<128xf32>, %arg287: tensor<1024x2048xf32>, %arg288: tensor<1024x1024xf32>, %arg289: tensor<1024xf32>, %arg290: tensor<3072x1024xf32>, %arg291: tensor<1024x3072xf32>, %arg292: tensor<1024x3072xf32>, %arg293: tensor<1024xf32>, %arg294: tensor<128xf32>, %arg295: tensor<1024x1024xf32>, %arg296: tensor<2048x1024xf32>, %arg297: tensor<128xf32>, %arg298: tensor<1024x2048xf32>, %arg299: tensor<1024x1024xf32>, %arg300: tensor<1024xf32>, %arg301: tensor<3072x1024xf32>, %arg302: tensor<1024x3072xf32>, %arg303: tensor<1024x3072xf32>, %arg304: tensor<1024xf32>, %arg305: tensor<128xf32>, %arg306: tensor<1024x1024xf32>, %arg307: tensor<2048x1024xf32>, %arg308: tensor<128xf32>, %arg309: tensor<1024x2048xf32>, %arg310: tensor<1024x1024xf32>, %arg311: tensor<1024xf32>, %arg312: tensor<3072x1024xf32>, %arg313: tensor<1024x3072xf32>, %arg314: tensor<1024x3072xf32>, %arg315: tensor<1024xf32>, %arg316: tensor<128xf32>, %arg317: tensor<1024x1024xf32>, %arg318: tensor<2048x1024xf32>, %arg319: tensor<128xf32>, %arg320: tensor<1024x2048xf32>, %arg321: tensor<1024x1024xf32>, %arg322: tensor<1024xf32>, %arg323: tensor<3072x1024xf32>, %arg324: tensor<1024x3072xf32>, %arg325: tensor<1024x3072xf32>, %arg326: tensor<1024xf32>, %arg327: tensor<128xf32>, %arg328: tensor<1024x1024xf32>, %arg329: tensor<2048x1024xf32>, %arg330: tensor<128xf32>, %arg331: tensor<1024x2048xf32>, %arg332: tensor<1024x1024xf32>, %arg333: tensor<1024xf32>, %arg334: tensor<3072x1024xf32>, %arg335: tensor<1024x3072xf32>, %arg336: tensor<1024x3072xf32>, %arg337: tensor<1024xf32>, %arg338: tensor<128xf32>, %arg339: tensor<1024x1024xf32>, %arg340: tensor<2048x1024xf32>, %arg341: tensor<128xf32>, %arg342: tensor<1024x2048xf32>, %arg343: tensor<1024x1024xf32>, %arg344: tensor<1024xf32>, %arg345: tensor<3072x1024xf32>, %arg346: tensor<1024x3072xf32>, %arg347: tensor<1024x3072xf32>, %arg348: tensor<1024xf32>, %arg349: tensor<128xf32>, %arg350: tensor<1024x1024xf32>, %arg351: tensor<2048x1024xf32>, %arg352: tensor<128xf32>, %arg353: tensor<1024x2048xf32>, %arg354: tensor<1024x1024xf32>, %arg355: tensor<1024xf32>, %arg356: tensor<3072x1024xf32>, %arg357: tensor<1024x3072xf32>, %arg358: tensor<1024x3072xf32>, %arg359: tensor<1024xf32>, %arg360: tensor<128xf32>, %arg361: tensor<1024x1024xf32>, %arg362: tensor<2048x1024xf32>, %arg363: tensor<128xf32>, %arg364: tensor<1024x2048xf32>, %arg365: tensor<1024x1024xf32>, %arg366: tensor<1024xf32>, %arg367: tensor<3072x1024xf32>, %arg368: tensor<1024x3072xf32>, %arg369: tensor<1024x3072xf32>, %arg370: tensor<1024xf32>, %arg371: tensor<128xf32>, %arg372: tensor<1024x1024xf32>, %arg373: tensor<2048x1024xf32>, %arg374: tensor<128xf32>, %arg375: tensor<1024x2048xf32>, %arg376: tensor<1024x1024xf32>, %arg377: tensor<1024xf32>, %arg378: tensor<3072x1024xf32>, %arg379: tensor<1024x3072xf32>, %arg380: tensor<1024x3072xf32>, %arg381: tensor<1024xf32>, %arg382: tensor<128xf32>, %arg383: tensor<1024x1024xf32>, %arg384: tensor<2048x1024xf32>, %arg385: tensor<128xf32>, %arg386: tensor<1024x2048xf32>, %arg387: tensor<1024x1024xf32>, %arg388: tensor<1024xf32>, %arg389: tensor<3072x1024xf32>, %arg390: tensor<1024x3072xf32>, %arg391: tensor<1024x3072xf32>, %arg392: tensor<1024xf32>, %arg393: tensor<128xf32>, %arg394: tensor<1024x1024xf32>, %arg395: tensor<2048x1024xf32>, %arg396: tensor<128xf32>, %arg397: tensor<1024x2048xf32>, %arg398: tensor<1024x1024xf32>, %arg399: tensor<1024xf32>, %arg400: tensor<3072x1024xf32>, %arg401: tensor<1024x3072xf32>, %arg402: tensor<1024x3072xf32>, %arg403: tensor<1024xf32>, %arg404: tensor<128xf32>, %arg405: tensor<1024x1024xf32>, %arg406: tensor<2048x1024xf32>, %arg407: tensor<128xf32>, %arg408: tensor<1024x2048xf32>, %arg409: tensor<1024x1024xf32>, %arg410: tensor<1024xf32>, %arg411: tensor<3072x1024xf32>, %arg412: tensor<1024x3072xf32>, %arg413: tensor<1024x3072xf32>, %arg414: tensor<1024xf32>, %arg415: tensor<128xf32>, %arg416: tensor<1024x1024xf32>, %arg417: tensor<2048x1024xf32>, %arg418: tensor<128xf32>, %arg419: tensor<1024x2048xf32>, %arg420: tensor<1024x1024xf32>, %arg421: tensor<1024xf32>, %arg422: tensor<1x8xui32>) -> (tensor<1x8x151936xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg422 : (tensor<1x8xui32>) -> tensor<1x8xi32>
    %1 = stablehlo.convert %arg112 : (tensor<151936x1024xf32>) -> tensor<151936x1024xbf16>
    %2 = call @_take(%1, %0) : (tensor<151936x1024xbf16>, tensor<1x8xi32>) -> tensor<1x8x1024xbf16>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<1x8xi1>
    %4 = call @cumsum(%3) : (tensor<1x8xi1>) -> tensor<1x8xi32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %6 = stablehlo.subtract %4, %5 : tensor<1x8xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %7 = call @clip(%6, %c_1) : (tensor<1x8xi32>, tensor<i32>) -> tensor<1x8xi32>
    %c_2 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i1>) -> tensor<128x128xi1>
    %9 = stablehlo.iota dim = 0 : tensor<128xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<128xi32>) -> tensor<128x1xi32>
    %11 = stablehlo.iota dim = 0 : tensor<128xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<128x1xi32>) -> tensor<128x128xi32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %15 = stablehlo.compare  GE, %13, %14,  SIGNED : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi1>
    %16 = stablehlo.or %8, %15 : tensor<128x128xi1>
    %17 = stablehlo.iota dim = 0 : tensor<128xi32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<128xi32>) -> tensor<128x1xi32>
    %19 = stablehlo.iota dim = 0 : tensor<128xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %22 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x1xi32>) -> tensor<128x128xi32>
    %23 = stablehlo.compare  LE, %21, %22,  SIGNED : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi1>
    %24 = stablehlo.and %16, %23 : tensor<128x128xi1>
    %25 = stablehlo.broadcast_in_dim %24, dims = [2, 3] : (tensor<128x128xi1>) -> tensor<1x1x128x128xi1>
    %26 = call @get_frequencies() : () -> tensor<40960x128xf32>
    %27 = stablehlo.convert %2 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<1x8x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.reduce(%28 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %cst_3 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %32 = stablehlo.divide %30, %31 : tensor<1x8x1xf32>
    %cst_4 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %34 = stablehlo.add %32, %33 : tensor<1x8x1xf32>
    %35 = stablehlo.rsqrt %34 : tensor<1x8x1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %37 = stablehlo.multiply %27, %36 : tensor<1x8x1024xf32>
    %38 = stablehlo.convert %37 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %39 = stablehlo.convert %arg113 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %40 = stablehlo.broadcast_in_dim %39, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %42 = stablehlo.multiply %41, %38 : tensor<1x8x1024xbf16>
    %43 = stablehlo.convert %arg0 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %44 = stablehlo.convert %arg1 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %45 = stablehlo.dot_general %42, %43, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %46 = stablehlo.dot_general %45, %44, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %47 = stablehlo.convert %arg122 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %48 = stablehlo.dot_general %42, %47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %49 = stablehlo.add %46, %48 : tensor<1x8x2048xbf16>
    %50 = stablehlo.convert %arg119 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %51 = stablehlo.dot_general %42, %50, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %52 = stablehlo.convert %arg2 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %53 = stablehlo.convert %arg3 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %54 = stablehlo.dot_general %42, %52, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %55 = stablehlo.dot_general %54, %53, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %56 = stablehlo.convert %arg123 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %57 = stablehlo.dot_general %42, %56, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %58 = stablehlo.add %55, %57 : tensor<1x8x1024xbf16>
    %59 = stablehlo.reshape %49 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %60 = stablehlo.convert %59 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %61 = stablehlo.multiply %60, %60 : tensor<1x8x16x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.reduce(%61 init: %cst_5) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %cst_6 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %65 = stablehlo.divide %63, %64 : tensor<1x8x16x1xf32>
    %66 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %67 = stablehlo.add %65, %66 : tensor<1x8x16x1xf32>
    %68 = stablehlo.rsqrt %67 : tensor<1x8x16x1xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %70 = stablehlo.multiply %60, %69 : tensor<1x8x16x128xf32>
    %71 = stablehlo.convert %70 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %72 = stablehlo.convert %arg121 : (tensor<128xf32>) -> tensor<128xbf16>
    %73 = stablehlo.broadcast_in_dim %72, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %75 = stablehlo.multiply %74, %71 : tensor<1x8x16x128xbf16>
    %76 = stablehlo.reshape %51 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %77 = stablehlo.convert %76 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %78 = stablehlo.multiply %77, %77 : tensor<1x8x8x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %81 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %82 = stablehlo.divide %80, %81 : tensor<1x8x8x1xf32>
    %83 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %84 = stablehlo.add %82, %83 : tensor<1x8x8x1xf32>
    %85 = stablehlo.rsqrt %84 : tensor<1x8x8x1xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %87 = stablehlo.multiply %77, %86 : tensor<1x8x8x128xf32>
    %88 = stablehlo.convert %87 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %89 = stablehlo.convert %arg118 : (tensor<128xf32>) -> tensor<128xbf16>
    %90 = stablehlo.broadcast_in_dim %89, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %92 = stablehlo.multiply %91, %88 : tensor<1x8x8x128xbf16>
    %93 = stablehlo.reshape %58 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %94 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %95 = stablehlo.compare  LT, %7, %94,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_9 = stablehlo.constant dense<40960> : tensor<i32>
    %96 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %97 = stablehlo.add %7, %96 : tensor<1x8xi32>
    %98 = stablehlo.select %95, %97, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %100 = "stablehlo.gather"(%26, %99) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %101 = stablehlo.slice %100 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %102 = stablehlo.slice %100 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %103 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %104 = stablehlo.convert %103 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %105 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %106 = stablehlo.convert %105 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %107 = stablehlo.slice %75 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %108 = stablehlo.slice %75 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %109 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %110 = stablehlo.multiply %107, %109 : tensor<1x8x16x64xbf16>
    %111 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %112 = stablehlo.multiply %108, %111 : tensor<1x8x16x64xbf16>
    %113 = stablehlo.subtract %110, %112 : tensor<1x8x16x64xbf16>
    %114 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %115 = stablehlo.multiply %108, %114 : tensor<1x8x16x64xbf16>
    %116 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %117 = stablehlo.multiply %107, %116 : tensor<1x8x16x64xbf16>
    %118 = stablehlo.add %115, %117 : tensor<1x8x16x64xbf16>
    %119 = stablehlo.concatenate %113, %118, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %120 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %121 = stablehlo.convert %120 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %122 = stablehlo.broadcast_in_dim %102, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %123 = stablehlo.convert %122 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %124 = stablehlo.slice %92 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %125 = stablehlo.slice %92 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %126 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %127 = stablehlo.multiply %124, %126 : tensor<1x8x8x64xbf16>
    %128 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %129 = stablehlo.multiply %125, %128 : tensor<1x8x8x64xbf16>
    %130 = stablehlo.subtract %127, %129 : tensor<1x8x8x64xbf16>
    %131 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %132 = stablehlo.multiply %125, %131 : tensor<1x8x8x64xbf16>
    %133 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %134 = stablehlo.multiply %124, %133 : tensor<1x8x8x64xbf16>
    %135 = stablehlo.add %132, %134 : tensor<1x8x8x64xbf16>
    %136 = stablehlo.concatenate %130, %135, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %137 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %138 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %139 = stablehlo.broadcast_in_dim %137, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %140 = stablehlo.and %139, %138 : tensor<1x1x8x8xi1>
    %141 = stablehlo.convert %140 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %142 = sdy.sharding_constraint %119 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %143 = sdy.sharding_constraint %136 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %144 = sdy.sharding_constraint %93 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %145 = sdy.sharding_constraint %141 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %146 = stablehlo.reshape %142 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %cst_10 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
    %147 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %148 = stablehlo.multiply %146, %147 : tensor<1x8x8x2x128xbf16>
    %149 = stablehlo.dot_general %143, %148, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %150 = stablehlo.transpose %149, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %152 = stablehlo.compare  NE, %145, %151,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %153 = stablehlo.convert %152 : tensor<1x1x8x8xi1>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %155 = stablehlo.reshape %154 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %cst_12 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
    %156 = call @_where_91(%155, %150, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %157 = stablehlo.convert %156 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_13 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %158 = stablehlo.reduce(%157 init: %cst_13) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %cst_14 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %159 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %160 = stablehlo.maximum %159, %158 : tensor<1x8x2x8xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %162 = stablehlo.broadcast_in_dim %161, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %163 = stablehlo.subtract %157, %162 : tensor<1x8x2x8x8xf32>
    %164 = stablehlo.exponential %163 : tensor<1x8x2x8x8xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %165 = stablehlo.reduce(%164 init: %cst_15) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %168 = stablehlo.divide %164, %167 : tensor<1x8x2x8x8xf32>
    %169 = stablehlo.convert %168 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %170 = stablehlo.dot_general %144, %169, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %171 = stablehlo.transpose %170, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %172 = stablehlo.reshape %171 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %173 = sdy.sharding_constraint %172 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %174 = stablehlo.reshape %173 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %175 = stablehlo.convert %arg120 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %176 = stablehlo.dot_general %174, %175, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %177 = stablehlo.add %2, %176 : tensor<1x8x1024xbf16>
    %178 = stablehlo.convert %177 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %179 = stablehlo.multiply %178, %178 : tensor<1x8x1024xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %180 = stablehlo.reduce(%179 init: %cst_16) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %182 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %183 = stablehlo.divide %181, %182 : tensor<1x8x1xf32>
    %184 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %185 = stablehlo.add %183, %184 : tensor<1x8x1xf32>
    %186 = stablehlo.rsqrt %185 : tensor<1x8x1xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %188 = stablehlo.multiply %178, %187 : tensor<1x8x1024xf32>
    %189 = stablehlo.convert %188 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %190 = stablehlo.convert %arg117 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %191 = stablehlo.broadcast_in_dim %190, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %193 = stablehlo.multiply %192, %189 : tensor<1x8x1024xbf16>
    %194 = stablehlo.convert %arg115 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %195 = stablehlo.dot_general %193, %194, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %196 = call @silu(%195) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %197 = stablehlo.convert %arg116 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %198 = stablehlo.dot_general %193, %197, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %199 = stablehlo.multiply %196, %198 : tensor<1x8x3072xbf16>
    %200 = stablehlo.convert %arg114 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %201 = stablehlo.dot_general %199, %200, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %202 = stablehlo.add %177, %201 : tensor<1x8x1024xbf16>
    %203 = stablehlo.convert %202 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %204 = stablehlo.multiply %203, %203 : tensor<1x8x1024xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %205 = stablehlo.reduce(%204 init: %cst_17) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %207 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %208 = stablehlo.divide %206, %207 : tensor<1x8x1xf32>
    %209 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %210 = stablehlo.add %208, %209 : tensor<1x8x1xf32>
    %211 = stablehlo.rsqrt %210 : tensor<1x8x1xf32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %213 = stablehlo.multiply %203, %212 : tensor<1x8x1024xf32>
    %214 = stablehlo.convert %213 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %215 = stablehlo.convert %arg124 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %216 = stablehlo.broadcast_in_dim %215, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %218 = stablehlo.multiply %217, %214 : tensor<1x8x1024xbf16>
    %219 = stablehlo.convert %arg4 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %220 = stablehlo.convert %arg5 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %221 = stablehlo.dot_general %218, %219, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %222 = stablehlo.dot_general %221, %220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %223 = stablehlo.convert %arg133 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %224 = stablehlo.dot_general %218, %223, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %225 = stablehlo.add %222, %224 : tensor<1x8x2048xbf16>
    %226 = stablehlo.convert %arg130 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %227 = stablehlo.dot_general %218, %226, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %228 = stablehlo.convert %arg6 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %229 = stablehlo.convert %arg7 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %230 = stablehlo.dot_general %218, %228, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %231 = stablehlo.dot_general %230, %229, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %232 = stablehlo.convert %arg134 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %233 = stablehlo.dot_general %218, %232, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %234 = stablehlo.add %231, %233 : tensor<1x8x1024xbf16>
    %235 = stablehlo.reshape %225 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %236 = stablehlo.convert %235 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %237 = stablehlo.multiply %236, %236 : tensor<1x8x16x128xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %238 = stablehlo.reduce(%237 init: %cst_18) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %240 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %241 = stablehlo.divide %239, %240 : tensor<1x8x16x1xf32>
    %242 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %243 = stablehlo.add %241, %242 : tensor<1x8x16x1xf32>
    %244 = stablehlo.rsqrt %243 : tensor<1x8x16x1xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %246 = stablehlo.multiply %236, %245 : tensor<1x8x16x128xf32>
    %247 = stablehlo.convert %246 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %248 = stablehlo.convert %arg132 : (tensor<128xf32>) -> tensor<128xbf16>
    %249 = stablehlo.broadcast_in_dim %248, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %251 = stablehlo.multiply %250, %247 : tensor<1x8x16x128xbf16>
    %252 = stablehlo.reshape %227 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %253 = stablehlo.convert %252 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %254 = stablehlo.multiply %253, %253 : tensor<1x8x8x128xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %255 = stablehlo.reduce(%254 init: %cst_19) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %257 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %258 = stablehlo.divide %256, %257 : tensor<1x8x8x1xf32>
    %259 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %260 = stablehlo.add %258, %259 : tensor<1x8x8x1xf32>
    %261 = stablehlo.rsqrt %260 : tensor<1x8x8x1xf32>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %263 = stablehlo.multiply %253, %262 : tensor<1x8x8x128xf32>
    %264 = stablehlo.convert %263 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %265 = stablehlo.convert %arg129 : (tensor<128xf32>) -> tensor<128xbf16>
    %266 = stablehlo.broadcast_in_dim %265, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %268 = stablehlo.multiply %267, %264 : tensor<1x8x8x128xbf16>
    %269 = stablehlo.reshape %234 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %270 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %271 = stablehlo.compare  LT, %7, %270,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %272 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %273 = stablehlo.add %7, %272 : tensor<1x8xi32>
    %274 = stablehlo.select %271, %273, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %276 = "stablehlo.gather"(%26, %275) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %277 = stablehlo.slice %276 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %278 = stablehlo.slice %276 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %279 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %280 = stablehlo.convert %279 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %281 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %282 = stablehlo.convert %281 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %283 = stablehlo.slice %251 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %284 = stablehlo.slice %251 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %285 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %286 = stablehlo.multiply %283, %285 : tensor<1x8x16x64xbf16>
    %287 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %288 = stablehlo.multiply %284, %287 : tensor<1x8x16x64xbf16>
    %289 = stablehlo.subtract %286, %288 : tensor<1x8x16x64xbf16>
    %290 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %291 = stablehlo.multiply %284, %290 : tensor<1x8x16x64xbf16>
    %292 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %293 = stablehlo.multiply %283, %292 : tensor<1x8x16x64xbf16>
    %294 = stablehlo.add %291, %293 : tensor<1x8x16x64xbf16>
    %295 = stablehlo.concatenate %289, %294, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %296 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %297 = stablehlo.convert %296 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %298 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %299 = stablehlo.convert %298 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %300 = stablehlo.slice %268 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %301 = stablehlo.slice %268 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %302 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %303 = stablehlo.multiply %300, %302 : tensor<1x8x8x64xbf16>
    %304 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %305 = stablehlo.multiply %301, %304 : tensor<1x8x8x64xbf16>
    %306 = stablehlo.subtract %303, %305 : tensor<1x8x8x64xbf16>
    %307 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %308 = stablehlo.multiply %301, %307 : tensor<1x8x8x64xbf16>
    %309 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %310 = stablehlo.multiply %300, %309 : tensor<1x8x8x64xbf16>
    %311 = stablehlo.add %308, %310 : tensor<1x8x8x64xbf16>
    %312 = stablehlo.concatenate %306, %311, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %313 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %314 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %315 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %316 = stablehlo.and %315, %314 : tensor<1x1x8x8xi1>
    %317 = stablehlo.convert %316 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %318 = sdy.sharding_constraint %295 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %319 = sdy.sharding_constraint %312 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %320 = sdy.sharding_constraint %269 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %321 = sdy.sharding_constraint %317 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %322 = stablehlo.reshape %318 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %323 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %324 = stablehlo.multiply %322, %323 : tensor<1x8x8x2x128xbf16>
    %325 = stablehlo.dot_general %319, %324, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %326 = stablehlo.transpose %325, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %327 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %328 = stablehlo.compare  NE, %321, %327,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %329 = stablehlo.convert %328 : tensor<1x1x8x8xi1>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %331 = stablehlo.reshape %330 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %332 = call @_where_91(%331, %326, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %333 = stablehlo.convert %332 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %334 = stablehlo.reduce(%333 init: %cst_21) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %335 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %336 = stablehlo.maximum %335, %334 : tensor<1x8x2x8xf32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %339 = stablehlo.subtract %333, %338 : tensor<1x8x2x8x8xf32>
    %340 = stablehlo.exponential %339 : tensor<1x8x2x8x8xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %341 = stablehlo.reduce(%340 init: %cst_22) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %342 = stablehlo.broadcast_in_dim %341, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %344 = stablehlo.divide %340, %343 : tensor<1x8x2x8x8xf32>
    %345 = stablehlo.convert %344 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %346 = stablehlo.dot_general %320, %345, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %347 = stablehlo.transpose %346, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %348 = stablehlo.reshape %347 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %349 = sdy.sharding_constraint %348 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %350 = stablehlo.reshape %349 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %351 = stablehlo.convert %arg131 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %352 = stablehlo.dot_general %350, %351, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %353 = stablehlo.add %202, %352 : tensor<1x8x1024xbf16>
    %354 = stablehlo.convert %353 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %355 = stablehlo.multiply %354, %354 : tensor<1x8x1024xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %356 = stablehlo.reduce(%355 init: %cst_23) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %358 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %359 = stablehlo.divide %357, %358 : tensor<1x8x1xf32>
    %360 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %361 = stablehlo.add %359, %360 : tensor<1x8x1xf32>
    %362 = stablehlo.rsqrt %361 : tensor<1x8x1xf32>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %364 = stablehlo.multiply %354, %363 : tensor<1x8x1024xf32>
    %365 = stablehlo.convert %364 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %366 = stablehlo.convert %arg128 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %367 = stablehlo.broadcast_in_dim %366, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %369 = stablehlo.multiply %368, %365 : tensor<1x8x1024xbf16>
    %370 = stablehlo.convert %arg126 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %371 = stablehlo.dot_general %369, %370, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %372 = call @silu(%371) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %373 = stablehlo.convert %arg127 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %374 = stablehlo.dot_general %369, %373, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %375 = stablehlo.multiply %372, %374 : tensor<1x8x3072xbf16>
    %376 = stablehlo.convert %arg125 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %377 = stablehlo.dot_general %375, %376, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %378 = stablehlo.add %353, %377 : tensor<1x8x1024xbf16>
    %379 = stablehlo.convert %378 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %380 = stablehlo.multiply %379, %379 : tensor<1x8x1024xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %381 = stablehlo.reduce(%380 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %383 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %384 = stablehlo.divide %382, %383 : tensor<1x8x1xf32>
    %385 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %386 = stablehlo.add %384, %385 : tensor<1x8x1xf32>
    %387 = stablehlo.rsqrt %386 : tensor<1x8x1xf32>
    %388 = stablehlo.broadcast_in_dim %387, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %389 = stablehlo.multiply %379, %388 : tensor<1x8x1024xf32>
    %390 = stablehlo.convert %389 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %391 = stablehlo.convert %arg135 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %392 = stablehlo.broadcast_in_dim %391, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %394 = stablehlo.multiply %393, %390 : tensor<1x8x1024xbf16>
    %395 = stablehlo.convert %arg8 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %396 = stablehlo.convert %arg9 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %397 = stablehlo.dot_general %394, %395, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %398 = stablehlo.dot_general %397, %396, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %399 = stablehlo.convert %arg144 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %400 = stablehlo.dot_general %394, %399, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %401 = stablehlo.add %398, %400 : tensor<1x8x2048xbf16>
    %402 = stablehlo.convert %arg141 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %403 = stablehlo.dot_general %394, %402, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %404 = stablehlo.convert %arg10 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %405 = stablehlo.convert %arg11 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %406 = stablehlo.dot_general %394, %404, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %407 = stablehlo.dot_general %406, %405, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %408 = stablehlo.convert %arg145 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %409 = stablehlo.dot_general %394, %408, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %410 = stablehlo.add %407, %409 : tensor<1x8x1024xbf16>
    %411 = stablehlo.reshape %401 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %412 = stablehlo.convert %411 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %413 = stablehlo.multiply %412, %412 : tensor<1x8x16x128xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %414 = stablehlo.reduce(%413 init: %cst_25) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %416 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %417 = stablehlo.divide %415, %416 : tensor<1x8x16x1xf32>
    %418 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %419 = stablehlo.add %417, %418 : tensor<1x8x16x1xf32>
    %420 = stablehlo.rsqrt %419 : tensor<1x8x16x1xf32>
    %421 = stablehlo.broadcast_in_dim %420, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %422 = stablehlo.multiply %412, %421 : tensor<1x8x16x128xf32>
    %423 = stablehlo.convert %422 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %424 = stablehlo.convert %arg143 : (tensor<128xf32>) -> tensor<128xbf16>
    %425 = stablehlo.broadcast_in_dim %424, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %427 = stablehlo.multiply %426, %423 : tensor<1x8x16x128xbf16>
    %428 = stablehlo.reshape %403 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %429 = stablehlo.convert %428 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %430 = stablehlo.multiply %429, %429 : tensor<1x8x8x128xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %431 = stablehlo.reduce(%430 init: %cst_26) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %433 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %434 = stablehlo.divide %432, %433 : tensor<1x8x8x1xf32>
    %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %436 = stablehlo.add %434, %435 : tensor<1x8x8x1xf32>
    %437 = stablehlo.rsqrt %436 : tensor<1x8x8x1xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %439 = stablehlo.multiply %429, %438 : tensor<1x8x8x128xf32>
    %440 = stablehlo.convert %439 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %441 = stablehlo.convert %arg140 : (tensor<128xf32>) -> tensor<128xbf16>
    %442 = stablehlo.broadcast_in_dim %441, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %444 = stablehlo.multiply %443, %440 : tensor<1x8x8x128xbf16>
    %445 = stablehlo.reshape %410 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %446 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %447 = stablehlo.compare  LT, %7, %446,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %448 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %449 = stablehlo.add %7, %448 : tensor<1x8xi32>
    %450 = stablehlo.select %447, %449, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %451 = stablehlo.broadcast_in_dim %450, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %452 = "stablehlo.gather"(%26, %451) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %453 = stablehlo.slice %452 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %454 = stablehlo.slice %452 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %455 = stablehlo.broadcast_in_dim %453, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %456 = stablehlo.convert %455 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %457 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %458 = stablehlo.convert %457 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %459 = stablehlo.slice %427 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %460 = stablehlo.slice %427 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %461 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %462 = stablehlo.multiply %459, %461 : tensor<1x8x16x64xbf16>
    %463 = stablehlo.broadcast_in_dim %458, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %464 = stablehlo.multiply %460, %463 : tensor<1x8x16x64xbf16>
    %465 = stablehlo.subtract %462, %464 : tensor<1x8x16x64xbf16>
    %466 = stablehlo.broadcast_in_dim %456, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %467 = stablehlo.multiply %460, %466 : tensor<1x8x16x64xbf16>
    %468 = stablehlo.broadcast_in_dim %458, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %469 = stablehlo.multiply %459, %468 : tensor<1x8x16x64xbf16>
    %470 = stablehlo.add %467, %469 : tensor<1x8x16x64xbf16>
    %471 = stablehlo.concatenate %465, %470, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %472 = stablehlo.broadcast_in_dim %453, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %473 = stablehlo.convert %472 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %474 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %475 = stablehlo.convert %474 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %476 = stablehlo.slice %444 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %477 = stablehlo.slice %444 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %478 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %479 = stablehlo.multiply %476, %478 : tensor<1x8x8x64xbf16>
    %480 = stablehlo.broadcast_in_dim %475, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %481 = stablehlo.multiply %477, %480 : tensor<1x8x8x64xbf16>
    %482 = stablehlo.subtract %479, %481 : tensor<1x8x8x64xbf16>
    %483 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %484 = stablehlo.multiply %477, %483 : tensor<1x8x8x64xbf16>
    %485 = stablehlo.broadcast_in_dim %475, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %486 = stablehlo.multiply %476, %485 : tensor<1x8x8x64xbf16>
    %487 = stablehlo.add %484, %486 : tensor<1x8x8x64xbf16>
    %488 = stablehlo.concatenate %482, %487, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %489 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %490 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %491 = stablehlo.broadcast_in_dim %489, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %492 = stablehlo.and %491, %490 : tensor<1x1x8x8xi1>
    %493 = stablehlo.convert %492 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %494 = sdy.sharding_constraint %471 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %495 = sdy.sharding_constraint %488 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %496 = sdy.sharding_constraint %445 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %497 = sdy.sharding_constraint %493 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %498 = stablehlo.reshape %494 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %499 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %500 = stablehlo.multiply %498, %499 : tensor<1x8x8x2x128xbf16>
    %501 = stablehlo.dot_general %495, %500, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %502 = stablehlo.transpose %501, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %504 = stablehlo.compare  NE, %497, %503,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %505 = stablehlo.convert %504 : tensor<1x1x8x8xi1>
    %506 = stablehlo.broadcast_in_dim %505, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %507 = stablehlo.reshape %506 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %508 = call @_where_91(%507, %502, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %509 = stablehlo.convert %508 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %510 = stablehlo.reduce(%509 init: %cst_28) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %511 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %512 = stablehlo.maximum %511, %510 : tensor<1x8x2x8xf32>
    %513 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %515 = stablehlo.subtract %509, %514 : tensor<1x8x2x8x8xf32>
    %516 = stablehlo.exponential %515 : tensor<1x8x2x8x8xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %517 = stablehlo.reduce(%516 init: %cst_29) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %518 = stablehlo.broadcast_in_dim %517, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %519 = stablehlo.broadcast_in_dim %518, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %520 = stablehlo.divide %516, %519 : tensor<1x8x2x8x8xf32>
    %521 = stablehlo.convert %520 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %522 = stablehlo.dot_general %496, %521, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %523 = stablehlo.transpose %522, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %524 = stablehlo.reshape %523 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %525 = sdy.sharding_constraint %524 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %526 = stablehlo.reshape %525 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %527 = stablehlo.convert %arg142 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %528 = stablehlo.dot_general %526, %527, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %529 = stablehlo.add %378, %528 : tensor<1x8x1024xbf16>
    %530 = stablehlo.convert %529 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %531 = stablehlo.multiply %530, %530 : tensor<1x8x1024xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %532 = stablehlo.reduce(%531 init: %cst_30) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %533 = stablehlo.broadcast_in_dim %532, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %534 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %535 = stablehlo.divide %533, %534 : tensor<1x8x1xf32>
    %536 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %537 = stablehlo.add %535, %536 : tensor<1x8x1xf32>
    %538 = stablehlo.rsqrt %537 : tensor<1x8x1xf32>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %540 = stablehlo.multiply %530, %539 : tensor<1x8x1024xf32>
    %541 = stablehlo.convert %540 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %542 = stablehlo.convert %arg139 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %543 = stablehlo.broadcast_in_dim %542, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %545 = stablehlo.multiply %544, %541 : tensor<1x8x1024xbf16>
    %546 = stablehlo.convert %arg137 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %547 = stablehlo.dot_general %545, %546, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %548 = call @silu(%547) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %549 = stablehlo.convert %arg138 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %550 = stablehlo.dot_general %545, %549, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %551 = stablehlo.multiply %548, %550 : tensor<1x8x3072xbf16>
    %552 = stablehlo.convert %arg136 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %553 = stablehlo.dot_general %551, %552, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %554 = stablehlo.add %529, %553 : tensor<1x8x1024xbf16>
    %555 = stablehlo.convert %554 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %556 = stablehlo.multiply %555, %555 : tensor<1x8x1024xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %557 = stablehlo.reduce(%556 init: %cst_31) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %558 = stablehlo.broadcast_in_dim %557, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %559 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %560 = stablehlo.divide %558, %559 : tensor<1x8x1xf32>
    %561 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %562 = stablehlo.add %560, %561 : tensor<1x8x1xf32>
    %563 = stablehlo.rsqrt %562 : tensor<1x8x1xf32>
    %564 = stablehlo.broadcast_in_dim %563, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %565 = stablehlo.multiply %555, %564 : tensor<1x8x1024xf32>
    %566 = stablehlo.convert %565 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %567 = stablehlo.convert %arg146 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %568 = stablehlo.broadcast_in_dim %567, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %569 = stablehlo.broadcast_in_dim %568, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %570 = stablehlo.multiply %569, %566 : tensor<1x8x1024xbf16>
    %571 = stablehlo.convert %arg12 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %572 = stablehlo.convert %arg13 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %573 = stablehlo.dot_general %570, %571, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %574 = stablehlo.dot_general %573, %572, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %575 = stablehlo.convert %arg155 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %576 = stablehlo.dot_general %570, %575, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %577 = stablehlo.add %574, %576 : tensor<1x8x2048xbf16>
    %578 = stablehlo.convert %arg152 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %579 = stablehlo.dot_general %570, %578, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %580 = stablehlo.convert %arg14 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %581 = stablehlo.convert %arg15 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %582 = stablehlo.dot_general %570, %580, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %583 = stablehlo.dot_general %582, %581, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %584 = stablehlo.convert %arg156 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %585 = stablehlo.dot_general %570, %584, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %586 = stablehlo.add %583, %585 : tensor<1x8x1024xbf16>
    %587 = stablehlo.reshape %577 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %588 = stablehlo.convert %587 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %589 = stablehlo.multiply %588, %588 : tensor<1x8x16x128xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %590 = stablehlo.reduce(%589 init: %cst_32) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %591 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %592 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %593 = stablehlo.divide %591, %592 : tensor<1x8x16x1xf32>
    %594 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %595 = stablehlo.add %593, %594 : tensor<1x8x16x1xf32>
    %596 = stablehlo.rsqrt %595 : tensor<1x8x16x1xf32>
    %597 = stablehlo.broadcast_in_dim %596, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %598 = stablehlo.multiply %588, %597 : tensor<1x8x16x128xf32>
    %599 = stablehlo.convert %598 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %600 = stablehlo.convert %arg154 : (tensor<128xf32>) -> tensor<128xbf16>
    %601 = stablehlo.broadcast_in_dim %600, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %602 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %603 = stablehlo.multiply %602, %599 : tensor<1x8x16x128xbf16>
    %604 = stablehlo.reshape %579 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %605 = stablehlo.convert %604 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %606 = stablehlo.multiply %605, %605 : tensor<1x8x8x128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %607 = stablehlo.reduce(%606 init: %cst_33) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %609 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %610 = stablehlo.divide %608, %609 : tensor<1x8x8x1xf32>
    %611 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %612 = stablehlo.add %610, %611 : tensor<1x8x8x1xf32>
    %613 = stablehlo.rsqrt %612 : tensor<1x8x8x1xf32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %615 = stablehlo.multiply %605, %614 : tensor<1x8x8x128xf32>
    %616 = stablehlo.convert %615 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %617 = stablehlo.convert %arg151 : (tensor<128xf32>) -> tensor<128xbf16>
    %618 = stablehlo.broadcast_in_dim %617, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %620 = stablehlo.multiply %619, %616 : tensor<1x8x8x128xbf16>
    %621 = stablehlo.reshape %586 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %622 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %623 = stablehlo.compare  LT, %7, %622,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %624 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %625 = stablehlo.add %7, %624 : tensor<1x8xi32>
    %626 = stablehlo.select %623, %625, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %628 = "stablehlo.gather"(%26, %627) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %629 = stablehlo.slice %628 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %630 = stablehlo.slice %628 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %631 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %632 = stablehlo.convert %631 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %633 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %634 = stablehlo.convert %633 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %635 = stablehlo.slice %603 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %636 = stablehlo.slice %603 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %637 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %638 = stablehlo.multiply %635, %637 : tensor<1x8x16x64xbf16>
    %639 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %640 = stablehlo.multiply %636, %639 : tensor<1x8x16x64xbf16>
    %641 = stablehlo.subtract %638, %640 : tensor<1x8x16x64xbf16>
    %642 = stablehlo.broadcast_in_dim %632, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %643 = stablehlo.multiply %636, %642 : tensor<1x8x16x64xbf16>
    %644 = stablehlo.broadcast_in_dim %634, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %645 = stablehlo.multiply %635, %644 : tensor<1x8x16x64xbf16>
    %646 = stablehlo.add %643, %645 : tensor<1x8x16x64xbf16>
    %647 = stablehlo.concatenate %641, %646, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %648 = stablehlo.broadcast_in_dim %629, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %649 = stablehlo.convert %648 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %650 = stablehlo.broadcast_in_dim %630, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %651 = stablehlo.convert %650 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %652 = stablehlo.slice %620 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %653 = stablehlo.slice %620 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %654 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %655 = stablehlo.multiply %652, %654 : tensor<1x8x8x64xbf16>
    %656 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %657 = stablehlo.multiply %653, %656 : tensor<1x8x8x64xbf16>
    %658 = stablehlo.subtract %655, %657 : tensor<1x8x8x64xbf16>
    %659 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %660 = stablehlo.multiply %653, %659 : tensor<1x8x8x64xbf16>
    %661 = stablehlo.broadcast_in_dim %651, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %662 = stablehlo.multiply %652, %661 : tensor<1x8x8x64xbf16>
    %663 = stablehlo.add %660, %662 : tensor<1x8x8x64xbf16>
    %664 = stablehlo.concatenate %658, %663, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %665 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %666 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %667 = stablehlo.broadcast_in_dim %665, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %668 = stablehlo.and %667, %666 : tensor<1x1x8x8xi1>
    %669 = stablehlo.convert %668 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %670 = sdy.sharding_constraint %647 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %671 = sdy.sharding_constraint %664 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %672 = sdy.sharding_constraint %621 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %673 = sdy.sharding_constraint %669 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %674 = stablehlo.reshape %670 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %675 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %676 = stablehlo.multiply %674, %675 : tensor<1x8x8x2x128xbf16>
    %677 = stablehlo.dot_general %671, %676, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %678 = stablehlo.transpose %677, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %679 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %680 = stablehlo.compare  NE, %673, %679,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %681 = stablehlo.convert %680 : tensor<1x1x8x8xi1>
    %682 = stablehlo.broadcast_in_dim %681, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %683 = stablehlo.reshape %682 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %684 = call @_where_91(%683, %678, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %685 = stablehlo.convert %684 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_35 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %686 = stablehlo.reduce(%685 init: %cst_35) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %687 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %688 = stablehlo.maximum %687, %686 : tensor<1x8x2x8xf32>
    %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %691 = stablehlo.subtract %685, %690 : tensor<1x8x2x8x8xf32>
    %692 = stablehlo.exponential %691 : tensor<1x8x2x8x8xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %693 = stablehlo.reduce(%692 init: %cst_36) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %696 = stablehlo.divide %692, %695 : tensor<1x8x2x8x8xf32>
    %697 = stablehlo.convert %696 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %698 = stablehlo.dot_general %672, %697, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %699 = stablehlo.transpose %698, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %700 = stablehlo.reshape %699 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %701 = sdy.sharding_constraint %700 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %702 = stablehlo.reshape %701 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %703 = stablehlo.convert %arg153 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %704 = stablehlo.dot_general %702, %703, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %705 = stablehlo.add %554, %704 : tensor<1x8x1024xbf16>
    %706 = stablehlo.convert %705 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %707 = stablehlo.multiply %706, %706 : tensor<1x8x1024xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %708 = stablehlo.reduce(%707 init: %cst_37) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %710 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %711 = stablehlo.divide %709, %710 : tensor<1x8x1xf32>
    %712 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %713 = stablehlo.add %711, %712 : tensor<1x8x1xf32>
    %714 = stablehlo.rsqrt %713 : tensor<1x8x1xf32>
    %715 = stablehlo.broadcast_in_dim %714, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %716 = stablehlo.multiply %706, %715 : tensor<1x8x1024xf32>
    %717 = stablehlo.convert %716 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %718 = stablehlo.convert %arg150 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %719 = stablehlo.broadcast_in_dim %718, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %720 = stablehlo.broadcast_in_dim %719, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %721 = stablehlo.multiply %720, %717 : tensor<1x8x1024xbf16>
    %722 = stablehlo.convert %arg148 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %723 = stablehlo.dot_general %721, %722, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %724 = call @silu(%723) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %725 = stablehlo.convert %arg149 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %726 = stablehlo.dot_general %721, %725, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %727 = stablehlo.multiply %724, %726 : tensor<1x8x3072xbf16>
    %728 = stablehlo.convert %arg147 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %729 = stablehlo.dot_general %727, %728, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %730 = stablehlo.add %705, %729 : tensor<1x8x1024xbf16>
    %731 = stablehlo.convert %730 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %732 = stablehlo.multiply %731, %731 : tensor<1x8x1024xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %733 = stablehlo.reduce(%732 init: %cst_38) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %734 = stablehlo.broadcast_in_dim %733, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %735 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %736 = stablehlo.divide %734, %735 : tensor<1x8x1xf32>
    %737 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %738 = stablehlo.add %736, %737 : tensor<1x8x1xf32>
    %739 = stablehlo.rsqrt %738 : tensor<1x8x1xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %741 = stablehlo.multiply %731, %740 : tensor<1x8x1024xf32>
    %742 = stablehlo.convert %741 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %743 = stablehlo.convert %arg157 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %744 = stablehlo.broadcast_in_dim %743, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %746 = stablehlo.multiply %745, %742 : tensor<1x8x1024xbf16>
    %747 = stablehlo.convert %arg16 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %748 = stablehlo.convert %arg17 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %749 = stablehlo.dot_general %746, %747, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %750 = stablehlo.dot_general %749, %748, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %751 = stablehlo.convert %arg166 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %752 = stablehlo.dot_general %746, %751, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %753 = stablehlo.add %750, %752 : tensor<1x8x2048xbf16>
    %754 = stablehlo.convert %arg163 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %755 = stablehlo.dot_general %746, %754, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %756 = stablehlo.convert %arg18 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %757 = stablehlo.convert %arg19 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %758 = stablehlo.dot_general %746, %756, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %759 = stablehlo.dot_general %758, %757, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %760 = stablehlo.convert %arg167 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %761 = stablehlo.dot_general %746, %760, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %762 = stablehlo.add %759, %761 : tensor<1x8x1024xbf16>
    %763 = stablehlo.reshape %753 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %764 = stablehlo.convert %763 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %765 = stablehlo.multiply %764, %764 : tensor<1x8x16x128xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %766 = stablehlo.reduce(%765 init: %cst_39) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %767 = stablehlo.broadcast_in_dim %766, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %768 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %769 = stablehlo.divide %767, %768 : tensor<1x8x16x1xf32>
    %770 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %771 = stablehlo.add %769, %770 : tensor<1x8x16x1xf32>
    %772 = stablehlo.rsqrt %771 : tensor<1x8x16x1xf32>
    %773 = stablehlo.broadcast_in_dim %772, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %774 = stablehlo.multiply %764, %773 : tensor<1x8x16x128xf32>
    %775 = stablehlo.convert %774 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %776 = stablehlo.convert %arg165 : (tensor<128xf32>) -> tensor<128xbf16>
    %777 = stablehlo.broadcast_in_dim %776, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %778 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %779 = stablehlo.multiply %778, %775 : tensor<1x8x16x128xbf16>
    %780 = stablehlo.reshape %755 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %781 = stablehlo.convert %780 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %782 = stablehlo.multiply %781, %781 : tensor<1x8x8x128xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %783 = stablehlo.reduce(%782 init: %cst_40) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %784 = stablehlo.broadcast_in_dim %783, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %785 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %786 = stablehlo.divide %784, %785 : tensor<1x8x8x1xf32>
    %787 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %788 = stablehlo.add %786, %787 : tensor<1x8x8x1xf32>
    %789 = stablehlo.rsqrt %788 : tensor<1x8x8x1xf32>
    %790 = stablehlo.broadcast_in_dim %789, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %791 = stablehlo.multiply %781, %790 : tensor<1x8x8x128xf32>
    %792 = stablehlo.convert %791 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %793 = stablehlo.convert %arg162 : (tensor<128xf32>) -> tensor<128xbf16>
    %794 = stablehlo.broadcast_in_dim %793, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %795 = stablehlo.broadcast_in_dim %794, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %796 = stablehlo.multiply %795, %792 : tensor<1x8x8x128xbf16>
    %797 = stablehlo.reshape %762 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %798 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %799 = stablehlo.compare  LT, %7, %798,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %800 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %801 = stablehlo.add %7, %800 : tensor<1x8xi32>
    %802 = stablehlo.select %799, %801, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %803 = stablehlo.broadcast_in_dim %802, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %804 = "stablehlo.gather"(%26, %803) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %805 = stablehlo.slice %804 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %806 = stablehlo.slice %804 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %807 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %808 = stablehlo.convert %807 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %809 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %810 = stablehlo.convert %809 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %811 = stablehlo.slice %779 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %812 = stablehlo.slice %779 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %813 = stablehlo.broadcast_in_dim %808, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %814 = stablehlo.multiply %811, %813 : tensor<1x8x16x64xbf16>
    %815 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %816 = stablehlo.multiply %812, %815 : tensor<1x8x16x64xbf16>
    %817 = stablehlo.subtract %814, %816 : tensor<1x8x16x64xbf16>
    %818 = stablehlo.broadcast_in_dim %808, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %819 = stablehlo.multiply %812, %818 : tensor<1x8x16x64xbf16>
    %820 = stablehlo.broadcast_in_dim %810, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %821 = stablehlo.multiply %811, %820 : tensor<1x8x16x64xbf16>
    %822 = stablehlo.add %819, %821 : tensor<1x8x16x64xbf16>
    %823 = stablehlo.concatenate %817, %822, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %824 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %825 = stablehlo.convert %824 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %826 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %827 = stablehlo.convert %826 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %828 = stablehlo.slice %796 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %829 = stablehlo.slice %796 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %830 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %831 = stablehlo.multiply %828, %830 : tensor<1x8x8x64xbf16>
    %832 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %833 = stablehlo.multiply %829, %832 : tensor<1x8x8x64xbf16>
    %834 = stablehlo.subtract %831, %833 : tensor<1x8x8x64xbf16>
    %835 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %836 = stablehlo.multiply %829, %835 : tensor<1x8x8x64xbf16>
    %837 = stablehlo.broadcast_in_dim %827, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %838 = stablehlo.multiply %828, %837 : tensor<1x8x8x64xbf16>
    %839 = stablehlo.add %836, %838 : tensor<1x8x8x64xbf16>
    %840 = stablehlo.concatenate %834, %839, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %841 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %842 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %843 = stablehlo.broadcast_in_dim %841, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %844 = stablehlo.and %843, %842 : tensor<1x1x8x8xi1>
    %845 = stablehlo.convert %844 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %846 = sdy.sharding_constraint %823 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %847 = sdy.sharding_constraint %840 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %848 = sdy.sharding_constraint %797 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %849 = sdy.sharding_constraint %845 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %850 = stablehlo.reshape %846 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %851 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %852 = stablehlo.multiply %850, %851 : tensor<1x8x8x2x128xbf16>
    %853 = stablehlo.dot_general %847, %852, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %854 = stablehlo.transpose %853, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %855 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %856 = stablehlo.compare  NE, %849, %855,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %857 = stablehlo.convert %856 : tensor<1x1x8x8xi1>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %859 = stablehlo.reshape %858 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %860 = call @_where_91(%859, %854, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %861 = stablehlo.convert %860 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_42 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %862 = stablehlo.reduce(%861 init: %cst_42) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %863 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %864 = stablehlo.maximum %863, %862 : tensor<1x8x2x8xf32>
    %865 = stablehlo.broadcast_in_dim %864, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %867 = stablehlo.subtract %861, %866 : tensor<1x8x2x8x8xf32>
    %868 = stablehlo.exponential %867 : tensor<1x8x2x8x8xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %869 = stablehlo.reduce(%868 init: %cst_43) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %870 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %872 = stablehlo.divide %868, %871 : tensor<1x8x2x8x8xf32>
    %873 = stablehlo.convert %872 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %874 = stablehlo.dot_general %848, %873, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %875 = stablehlo.transpose %874, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %876 = stablehlo.reshape %875 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %877 = sdy.sharding_constraint %876 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %878 = stablehlo.reshape %877 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %879 = stablehlo.convert %arg164 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %880 = stablehlo.dot_general %878, %879, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %881 = stablehlo.add %730, %880 : tensor<1x8x1024xbf16>
    %882 = stablehlo.convert %881 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %883 = stablehlo.multiply %882, %882 : tensor<1x8x1024xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %884 = stablehlo.reduce(%883 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %885 = stablehlo.broadcast_in_dim %884, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %886 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %887 = stablehlo.divide %885, %886 : tensor<1x8x1xf32>
    %888 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %889 = stablehlo.add %887, %888 : tensor<1x8x1xf32>
    %890 = stablehlo.rsqrt %889 : tensor<1x8x1xf32>
    %891 = stablehlo.broadcast_in_dim %890, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %892 = stablehlo.multiply %882, %891 : tensor<1x8x1024xf32>
    %893 = stablehlo.convert %892 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %894 = stablehlo.convert %arg161 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %895 = stablehlo.broadcast_in_dim %894, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %896 = stablehlo.broadcast_in_dim %895, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %897 = stablehlo.multiply %896, %893 : tensor<1x8x1024xbf16>
    %898 = stablehlo.convert %arg159 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %899 = stablehlo.dot_general %897, %898, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %900 = call @silu(%899) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %901 = stablehlo.convert %arg160 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %902 = stablehlo.dot_general %897, %901, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %903 = stablehlo.multiply %900, %902 : tensor<1x8x3072xbf16>
    %904 = stablehlo.convert %arg158 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %905 = stablehlo.dot_general %903, %904, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %906 = stablehlo.add %881, %905 : tensor<1x8x1024xbf16>
    %907 = stablehlo.convert %906 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %908 = stablehlo.multiply %907, %907 : tensor<1x8x1024xf32>
    %cst_45 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %909 = stablehlo.reduce(%908 init: %cst_45) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %910 = stablehlo.broadcast_in_dim %909, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %911 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %912 = stablehlo.divide %910, %911 : tensor<1x8x1xf32>
    %913 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %914 = stablehlo.add %912, %913 : tensor<1x8x1xf32>
    %915 = stablehlo.rsqrt %914 : tensor<1x8x1xf32>
    %916 = stablehlo.broadcast_in_dim %915, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %917 = stablehlo.multiply %907, %916 : tensor<1x8x1024xf32>
    %918 = stablehlo.convert %917 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %919 = stablehlo.convert %arg168 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %920 = stablehlo.broadcast_in_dim %919, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %922 = stablehlo.multiply %921, %918 : tensor<1x8x1024xbf16>
    %923 = stablehlo.convert %arg20 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %924 = stablehlo.convert %arg21 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %925 = stablehlo.dot_general %922, %923, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %926 = stablehlo.dot_general %925, %924, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %927 = stablehlo.convert %arg177 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %928 = stablehlo.dot_general %922, %927, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %929 = stablehlo.add %926, %928 : tensor<1x8x2048xbf16>
    %930 = stablehlo.convert %arg174 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %931 = stablehlo.dot_general %922, %930, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %932 = stablehlo.convert %arg22 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %933 = stablehlo.convert %arg23 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %934 = stablehlo.dot_general %922, %932, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %935 = stablehlo.dot_general %934, %933, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %936 = stablehlo.convert %arg178 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %937 = stablehlo.dot_general %922, %936, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %938 = stablehlo.add %935, %937 : tensor<1x8x1024xbf16>
    %939 = stablehlo.reshape %929 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %940 = stablehlo.convert %939 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %941 = stablehlo.multiply %940, %940 : tensor<1x8x16x128xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %942 = stablehlo.reduce(%941 init: %cst_46) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %943 = stablehlo.broadcast_in_dim %942, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %944 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %945 = stablehlo.divide %943, %944 : tensor<1x8x16x1xf32>
    %946 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %947 = stablehlo.add %945, %946 : tensor<1x8x16x1xf32>
    %948 = stablehlo.rsqrt %947 : tensor<1x8x16x1xf32>
    %949 = stablehlo.broadcast_in_dim %948, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %950 = stablehlo.multiply %940, %949 : tensor<1x8x16x128xf32>
    %951 = stablehlo.convert %950 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %952 = stablehlo.convert %arg176 : (tensor<128xf32>) -> tensor<128xbf16>
    %953 = stablehlo.broadcast_in_dim %952, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %954 = stablehlo.broadcast_in_dim %953, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %955 = stablehlo.multiply %954, %951 : tensor<1x8x16x128xbf16>
    %956 = stablehlo.reshape %931 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %957 = stablehlo.convert %956 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %958 = stablehlo.multiply %957, %957 : tensor<1x8x8x128xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %959 = stablehlo.reduce(%958 init: %cst_47) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %960 = stablehlo.broadcast_in_dim %959, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %961 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %962 = stablehlo.divide %960, %961 : tensor<1x8x8x1xf32>
    %963 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %964 = stablehlo.add %962, %963 : tensor<1x8x8x1xf32>
    %965 = stablehlo.rsqrt %964 : tensor<1x8x8x1xf32>
    %966 = stablehlo.broadcast_in_dim %965, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %967 = stablehlo.multiply %957, %966 : tensor<1x8x8x128xf32>
    %968 = stablehlo.convert %967 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %969 = stablehlo.convert %arg173 : (tensor<128xf32>) -> tensor<128xbf16>
    %970 = stablehlo.broadcast_in_dim %969, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %971 = stablehlo.broadcast_in_dim %970, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %972 = stablehlo.multiply %971, %968 : tensor<1x8x8x128xbf16>
    %973 = stablehlo.reshape %938 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %974 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %975 = stablehlo.compare  LT, %7, %974,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %976 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %977 = stablehlo.add %7, %976 : tensor<1x8xi32>
    %978 = stablehlo.select %975, %977, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %979 = stablehlo.broadcast_in_dim %978, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %980 = "stablehlo.gather"(%26, %979) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %981 = stablehlo.slice %980 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %982 = stablehlo.slice %980 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %983 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %984 = stablehlo.convert %983 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %985 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %986 = stablehlo.convert %985 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %987 = stablehlo.slice %955 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %988 = stablehlo.slice %955 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %989 = stablehlo.broadcast_in_dim %984, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %990 = stablehlo.multiply %987, %989 : tensor<1x8x16x64xbf16>
    %991 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %992 = stablehlo.multiply %988, %991 : tensor<1x8x16x64xbf16>
    %993 = stablehlo.subtract %990, %992 : tensor<1x8x16x64xbf16>
    %994 = stablehlo.broadcast_in_dim %984, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %995 = stablehlo.multiply %988, %994 : tensor<1x8x16x64xbf16>
    %996 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %997 = stablehlo.multiply %987, %996 : tensor<1x8x16x64xbf16>
    %998 = stablehlo.add %995, %997 : tensor<1x8x16x64xbf16>
    %999 = stablehlo.concatenate %993, %998, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1000 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1001 = stablehlo.convert %1000 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1002 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1003 = stablehlo.convert %1002 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1004 = stablehlo.slice %972 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1005 = stablehlo.slice %972 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1006 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1007 = stablehlo.multiply %1004, %1006 : tensor<1x8x8x64xbf16>
    %1008 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1009 = stablehlo.multiply %1005, %1008 : tensor<1x8x8x64xbf16>
    %1010 = stablehlo.subtract %1007, %1009 : tensor<1x8x8x64xbf16>
    %1011 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1012 = stablehlo.multiply %1005, %1011 : tensor<1x8x8x64xbf16>
    %1013 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1014 = stablehlo.multiply %1004, %1013 : tensor<1x8x8x64xbf16>
    %1015 = stablehlo.add %1012, %1014 : tensor<1x8x8x64xbf16>
    %1016 = stablehlo.concatenate %1010, %1015, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1017 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1018 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1019 = stablehlo.broadcast_in_dim %1017, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1020 = stablehlo.and %1019, %1018 : tensor<1x1x8x8xi1>
    %1021 = stablehlo.convert %1020 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1022 = sdy.sharding_constraint %999 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1023 = sdy.sharding_constraint %1016 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1024 = sdy.sharding_constraint %973 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1025 = sdy.sharding_constraint %1021 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1026 = stablehlo.reshape %1022 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1027 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1028 = stablehlo.multiply %1026, %1027 : tensor<1x8x8x2x128xbf16>
    %1029 = stablehlo.dot_general %1023, %1028, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1030 = stablehlo.transpose %1029, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1031 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1032 = stablehlo.compare  NE, %1025, %1031,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1033 = stablehlo.convert %1032 : tensor<1x1x8x8xi1>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1035 = stablehlo.reshape %1034 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1036 = call @_where_91(%1035, %1030, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1037 = stablehlo.convert %1036 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_49 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1038 = stablehlo.reduce(%1037 init: %cst_49) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1039 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1040 = stablehlo.maximum %1039, %1038 : tensor<1x8x2x8xf32>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1042 = stablehlo.broadcast_in_dim %1041, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1043 = stablehlo.subtract %1037, %1042 : tensor<1x8x2x8x8xf32>
    %1044 = stablehlo.exponential %1043 : tensor<1x8x2x8x8xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1045 = stablehlo.reduce(%1044 init: %cst_50) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1046 = stablehlo.broadcast_in_dim %1045, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1047 = stablehlo.broadcast_in_dim %1046, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1048 = stablehlo.divide %1044, %1047 : tensor<1x8x2x8x8xf32>
    %1049 = stablehlo.convert %1048 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1050 = stablehlo.dot_general %1024, %1049, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1051 = stablehlo.transpose %1050, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1052 = stablehlo.reshape %1051 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1053 = sdy.sharding_constraint %1052 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1054 = stablehlo.reshape %1053 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1055 = stablehlo.convert %arg175 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1056 = stablehlo.dot_general %1054, %1055, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1057 = stablehlo.add %906, %1056 : tensor<1x8x1024xbf16>
    %1058 = stablehlo.convert %1057 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1059 = stablehlo.multiply %1058, %1058 : tensor<1x8x1024xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1060 = stablehlo.reduce(%1059 init: %cst_51) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1061 = stablehlo.broadcast_in_dim %1060, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1062 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1063 = stablehlo.divide %1061, %1062 : tensor<1x8x1xf32>
    %1064 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1065 = stablehlo.add %1063, %1064 : tensor<1x8x1xf32>
    %1066 = stablehlo.rsqrt %1065 : tensor<1x8x1xf32>
    %1067 = stablehlo.broadcast_in_dim %1066, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1068 = stablehlo.multiply %1058, %1067 : tensor<1x8x1024xf32>
    %1069 = stablehlo.convert %1068 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1070 = stablehlo.convert %arg172 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1071 = stablehlo.broadcast_in_dim %1070, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1072 = stablehlo.broadcast_in_dim %1071, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1073 = stablehlo.multiply %1072, %1069 : tensor<1x8x1024xbf16>
    %1074 = stablehlo.convert %arg170 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1075 = stablehlo.dot_general %1073, %1074, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1076 = call @silu(%1075) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1077 = stablehlo.convert %arg171 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1078 = stablehlo.dot_general %1073, %1077, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1079 = stablehlo.multiply %1076, %1078 : tensor<1x8x3072xbf16>
    %1080 = stablehlo.convert %arg169 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1081 = stablehlo.dot_general %1079, %1080, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1082 = stablehlo.add %1057, %1081 : tensor<1x8x1024xbf16>
    %1083 = stablehlo.convert %1082 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1084 = stablehlo.multiply %1083, %1083 : tensor<1x8x1024xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1085 = stablehlo.reduce(%1084 init: %cst_52) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1086 = stablehlo.broadcast_in_dim %1085, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1087 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1088 = stablehlo.divide %1086, %1087 : tensor<1x8x1xf32>
    %1089 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1090 = stablehlo.add %1088, %1089 : tensor<1x8x1xf32>
    %1091 = stablehlo.rsqrt %1090 : tensor<1x8x1xf32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1093 = stablehlo.multiply %1083, %1092 : tensor<1x8x1024xf32>
    %1094 = stablehlo.convert %1093 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1095 = stablehlo.convert %arg179 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1096 = stablehlo.broadcast_in_dim %1095, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1098 = stablehlo.multiply %1097, %1094 : tensor<1x8x1024xbf16>
    %1099 = stablehlo.convert %arg24 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1100 = stablehlo.convert %arg25 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1101 = stablehlo.dot_general %1098, %1099, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1102 = stablehlo.dot_general %1101, %1100, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1103 = stablehlo.convert %arg188 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1104 = stablehlo.dot_general %1098, %1103, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1105 = stablehlo.add %1102, %1104 : tensor<1x8x2048xbf16>
    %1106 = stablehlo.convert %arg185 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1107 = stablehlo.dot_general %1098, %1106, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1108 = stablehlo.convert %arg26 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1109 = stablehlo.convert %arg27 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1110 = stablehlo.dot_general %1098, %1108, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1111 = stablehlo.dot_general %1110, %1109, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1112 = stablehlo.convert %arg189 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1113 = stablehlo.dot_general %1098, %1112, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1114 = stablehlo.add %1111, %1113 : tensor<1x8x1024xbf16>
    %1115 = stablehlo.reshape %1105 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1116 = stablehlo.convert %1115 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1117 = stablehlo.multiply %1116, %1116 : tensor<1x8x16x128xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1118 = stablehlo.reduce(%1117 init: %cst_53) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1119 = stablehlo.broadcast_in_dim %1118, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1120 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1121 = stablehlo.divide %1119, %1120 : tensor<1x8x16x1xf32>
    %1122 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1123 = stablehlo.add %1121, %1122 : tensor<1x8x16x1xf32>
    %1124 = stablehlo.rsqrt %1123 : tensor<1x8x16x1xf32>
    %1125 = stablehlo.broadcast_in_dim %1124, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1126 = stablehlo.multiply %1116, %1125 : tensor<1x8x16x128xf32>
    %1127 = stablehlo.convert %1126 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1128 = stablehlo.convert %arg187 : (tensor<128xf32>) -> tensor<128xbf16>
    %1129 = stablehlo.broadcast_in_dim %1128, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1130 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1131 = stablehlo.multiply %1130, %1127 : tensor<1x8x16x128xbf16>
    %1132 = stablehlo.reshape %1107 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1133 = stablehlo.convert %1132 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1134 = stablehlo.multiply %1133, %1133 : tensor<1x8x8x128xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1135 = stablehlo.reduce(%1134 init: %cst_54) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1136 = stablehlo.broadcast_in_dim %1135, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1137 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1138 = stablehlo.divide %1136, %1137 : tensor<1x8x8x1xf32>
    %1139 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1140 = stablehlo.add %1138, %1139 : tensor<1x8x8x1xf32>
    %1141 = stablehlo.rsqrt %1140 : tensor<1x8x8x1xf32>
    %1142 = stablehlo.broadcast_in_dim %1141, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1143 = stablehlo.multiply %1133, %1142 : tensor<1x8x8x128xf32>
    %1144 = stablehlo.convert %1143 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1145 = stablehlo.convert %arg184 : (tensor<128xf32>) -> tensor<128xbf16>
    %1146 = stablehlo.broadcast_in_dim %1145, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1147 = stablehlo.broadcast_in_dim %1146, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1148 = stablehlo.multiply %1147, %1144 : tensor<1x8x8x128xbf16>
    %1149 = stablehlo.reshape %1114 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1150 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1151 = stablehlo.compare  LT, %7, %1150,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1152 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1153 = stablehlo.add %7, %1152 : tensor<1x8xi32>
    %1154 = stablehlo.select %1151, %1153, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1156 = "stablehlo.gather"(%26, %1155) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1157 = stablehlo.slice %1156 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1158 = stablehlo.slice %1156 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1159 = stablehlo.broadcast_in_dim %1157, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1160 = stablehlo.convert %1159 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1161 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1162 = stablehlo.convert %1161 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1163 = stablehlo.slice %1131 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1164 = stablehlo.slice %1131 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1165 = stablehlo.broadcast_in_dim %1160, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1166 = stablehlo.multiply %1163, %1165 : tensor<1x8x16x64xbf16>
    %1167 = stablehlo.broadcast_in_dim %1162, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1168 = stablehlo.multiply %1164, %1167 : tensor<1x8x16x64xbf16>
    %1169 = stablehlo.subtract %1166, %1168 : tensor<1x8x16x64xbf16>
    %1170 = stablehlo.broadcast_in_dim %1160, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1171 = stablehlo.multiply %1164, %1170 : tensor<1x8x16x64xbf16>
    %1172 = stablehlo.broadcast_in_dim %1162, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1173 = stablehlo.multiply %1163, %1172 : tensor<1x8x16x64xbf16>
    %1174 = stablehlo.add %1171, %1173 : tensor<1x8x16x64xbf16>
    %1175 = stablehlo.concatenate %1169, %1174, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1176 = stablehlo.broadcast_in_dim %1157, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1177 = stablehlo.convert %1176 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1178 = stablehlo.broadcast_in_dim %1158, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1179 = stablehlo.convert %1178 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1180 = stablehlo.slice %1148 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1181 = stablehlo.slice %1148 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1182 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1183 = stablehlo.multiply %1180, %1182 : tensor<1x8x8x64xbf16>
    %1184 = stablehlo.broadcast_in_dim %1179, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1185 = stablehlo.multiply %1181, %1184 : tensor<1x8x8x64xbf16>
    %1186 = stablehlo.subtract %1183, %1185 : tensor<1x8x8x64xbf16>
    %1187 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1188 = stablehlo.multiply %1181, %1187 : tensor<1x8x8x64xbf16>
    %1189 = stablehlo.broadcast_in_dim %1179, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1190 = stablehlo.multiply %1180, %1189 : tensor<1x8x8x64xbf16>
    %1191 = stablehlo.add %1188, %1190 : tensor<1x8x8x64xbf16>
    %1192 = stablehlo.concatenate %1186, %1191, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1193 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1194 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1195 = stablehlo.broadcast_in_dim %1193, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1196 = stablehlo.and %1195, %1194 : tensor<1x1x8x8xi1>
    %1197 = stablehlo.convert %1196 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1198 = sdy.sharding_constraint %1175 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1199 = sdy.sharding_constraint %1192 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1200 = sdy.sharding_constraint %1149 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1201 = sdy.sharding_constraint %1197 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1202 = stablehlo.reshape %1198 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1203 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1204 = stablehlo.multiply %1202, %1203 : tensor<1x8x8x2x128xbf16>
    %1205 = stablehlo.dot_general %1199, %1204, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1206 = stablehlo.transpose %1205, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1207 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1208 = stablehlo.compare  NE, %1201, %1207,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1209 = stablehlo.convert %1208 : tensor<1x1x8x8xi1>
    %1210 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1211 = stablehlo.reshape %1210 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1212 = call @_where_91(%1211, %1206, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1213 = stablehlo.convert %1212 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_56 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1214 = stablehlo.reduce(%1213 init: %cst_56) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1215 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1216 = stablehlo.maximum %1215, %1214 : tensor<1x8x2x8xf32>
    %1217 = stablehlo.broadcast_in_dim %1216, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1218 = stablehlo.broadcast_in_dim %1217, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1219 = stablehlo.subtract %1213, %1218 : tensor<1x8x2x8x8xf32>
    %1220 = stablehlo.exponential %1219 : tensor<1x8x2x8x8xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1221 = stablehlo.reduce(%1220 init: %cst_57) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1222 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1223 = stablehlo.broadcast_in_dim %1222, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1224 = stablehlo.divide %1220, %1223 : tensor<1x8x2x8x8xf32>
    %1225 = stablehlo.convert %1224 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1226 = stablehlo.dot_general %1200, %1225, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1227 = stablehlo.transpose %1226, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1228 = stablehlo.reshape %1227 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1229 = sdy.sharding_constraint %1228 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1230 = stablehlo.reshape %1229 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1231 = stablehlo.convert %arg186 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1232 = stablehlo.dot_general %1230, %1231, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1233 = stablehlo.add %1082, %1232 : tensor<1x8x1024xbf16>
    %1234 = stablehlo.convert %1233 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1235 = stablehlo.multiply %1234, %1234 : tensor<1x8x1024xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1236 = stablehlo.reduce(%1235 init: %cst_58) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1237 = stablehlo.broadcast_in_dim %1236, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1238 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1239 = stablehlo.divide %1237, %1238 : tensor<1x8x1xf32>
    %1240 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1241 = stablehlo.add %1239, %1240 : tensor<1x8x1xf32>
    %1242 = stablehlo.rsqrt %1241 : tensor<1x8x1xf32>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1244 = stablehlo.multiply %1234, %1243 : tensor<1x8x1024xf32>
    %1245 = stablehlo.convert %1244 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1246 = stablehlo.convert %arg183 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1247 = stablehlo.broadcast_in_dim %1246, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1248 = stablehlo.broadcast_in_dim %1247, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1249 = stablehlo.multiply %1248, %1245 : tensor<1x8x1024xbf16>
    %1250 = stablehlo.convert %arg181 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1251 = stablehlo.dot_general %1249, %1250, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1252 = call @silu(%1251) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1253 = stablehlo.convert %arg182 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1254 = stablehlo.dot_general %1249, %1253, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1255 = stablehlo.multiply %1252, %1254 : tensor<1x8x3072xbf16>
    %1256 = stablehlo.convert %arg180 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1257 = stablehlo.dot_general %1255, %1256, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1258 = stablehlo.add %1233, %1257 : tensor<1x8x1024xbf16>
    %1259 = stablehlo.convert %1258 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1260 = stablehlo.multiply %1259, %1259 : tensor<1x8x1024xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1261 = stablehlo.reduce(%1260 init: %cst_59) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1262 = stablehlo.broadcast_in_dim %1261, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1263 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1264 = stablehlo.divide %1262, %1263 : tensor<1x8x1xf32>
    %1265 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1266 = stablehlo.add %1264, %1265 : tensor<1x8x1xf32>
    %1267 = stablehlo.rsqrt %1266 : tensor<1x8x1xf32>
    %1268 = stablehlo.broadcast_in_dim %1267, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1269 = stablehlo.multiply %1259, %1268 : tensor<1x8x1024xf32>
    %1270 = stablehlo.convert %1269 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1271 = stablehlo.convert %arg190 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1272 = stablehlo.broadcast_in_dim %1271, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1273 = stablehlo.broadcast_in_dim %1272, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1274 = stablehlo.multiply %1273, %1270 : tensor<1x8x1024xbf16>
    %1275 = stablehlo.convert %arg28 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1276 = stablehlo.convert %arg29 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1277 = stablehlo.dot_general %1274, %1275, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1278 = stablehlo.dot_general %1277, %1276, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1279 = stablehlo.convert %arg199 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1280 = stablehlo.dot_general %1274, %1279, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1281 = stablehlo.add %1278, %1280 : tensor<1x8x2048xbf16>
    %1282 = stablehlo.convert %arg196 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1283 = stablehlo.dot_general %1274, %1282, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1284 = stablehlo.convert %arg30 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1285 = stablehlo.convert %arg31 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1286 = stablehlo.dot_general %1274, %1284, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1287 = stablehlo.dot_general %1286, %1285, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1288 = stablehlo.convert %arg200 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1289 = stablehlo.dot_general %1274, %1288, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1290 = stablehlo.add %1287, %1289 : tensor<1x8x1024xbf16>
    %1291 = stablehlo.reshape %1281 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1292 = stablehlo.convert %1291 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1293 = stablehlo.multiply %1292, %1292 : tensor<1x8x16x128xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1294 = stablehlo.reduce(%1293 init: %cst_60) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1295 = stablehlo.broadcast_in_dim %1294, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1296 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1297 = stablehlo.divide %1295, %1296 : tensor<1x8x16x1xf32>
    %1298 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1299 = stablehlo.add %1297, %1298 : tensor<1x8x16x1xf32>
    %1300 = stablehlo.rsqrt %1299 : tensor<1x8x16x1xf32>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1302 = stablehlo.multiply %1292, %1301 : tensor<1x8x16x128xf32>
    %1303 = stablehlo.convert %1302 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1304 = stablehlo.convert %arg198 : (tensor<128xf32>) -> tensor<128xbf16>
    %1305 = stablehlo.broadcast_in_dim %1304, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1307 = stablehlo.multiply %1306, %1303 : tensor<1x8x16x128xbf16>
    %1308 = stablehlo.reshape %1283 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1309 = stablehlo.convert %1308 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1310 = stablehlo.multiply %1309, %1309 : tensor<1x8x8x128xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1311 = stablehlo.reduce(%1310 init: %cst_61) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1312 = stablehlo.broadcast_in_dim %1311, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1313 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1314 = stablehlo.divide %1312, %1313 : tensor<1x8x8x1xf32>
    %1315 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1316 = stablehlo.add %1314, %1315 : tensor<1x8x8x1xf32>
    %1317 = stablehlo.rsqrt %1316 : tensor<1x8x8x1xf32>
    %1318 = stablehlo.broadcast_in_dim %1317, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1319 = stablehlo.multiply %1309, %1318 : tensor<1x8x8x128xf32>
    %1320 = stablehlo.convert %1319 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1321 = stablehlo.convert %arg195 : (tensor<128xf32>) -> tensor<128xbf16>
    %1322 = stablehlo.broadcast_in_dim %1321, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1324 = stablehlo.multiply %1323, %1320 : tensor<1x8x8x128xbf16>
    %1325 = stablehlo.reshape %1290 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1326 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1327 = stablehlo.compare  LT, %7, %1326,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1328 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1329 = stablehlo.add %7, %1328 : tensor<1x8xi32>
    %1330 = stablehlo.select %1327, %1329, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1332 = "stablehlo.gather"(%26, %1331) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1333 = stablehlo.slice %1332 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1334 = stablehlo.slice %1332 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1335 = stablehlo.broadcast_in_dim %1333, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1336 = stablehlo.convert %1335 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1337 = stablehlo.broadcast_in_dim %1334, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1338 = stablehlo.convert %1337 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1339 = stablehlo.slice %1307 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1340 = stablehlo.slice %1307 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1341 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1342 = stablehlo.multiply %1339, %1341 : tensor<1x8x16x64xbf16>
    %1343 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1344 = stablehlo.multiply %1340, %1343 : tensor<1x8x16x64xbf16>
    %1345 = stablehlo.subtract %1342, %1344 : tensor<1x8x16x64xbf16>
    %1346 = stablehlo.broadcast_in_dim %1336, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1347 = stablehlo.multiply %1340, %1346 : tensor<1x8x16x64xbf16>
    %1348 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1349 = stablehlo.multiply %1339, %1348 : tensor<1x8x16x64xbf16>
    %1350 = stablehlo.add %1347, %1349 : tensor<1x8x16x64xbf16>
    %1351 = stablehlo.concatenate %1345, %1350, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1352 = stablehlo.broadcast_in_dim %1333, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1353 = stablehlo.convert %1352 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1354 = stablehlo.broadcast_in_dim %1334, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1355 = stablehlo.convert %1354 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1356 = stablehlo.slice %1324 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1357 = stablehlo.slice %1324 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1358 = stablehlo.broadcast_in_dim %1353, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1359 = stablehlo.multiply %1356, %1358 : tensor<1x8x8x64xbf16>
    %1360 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1361 = stablehlo.multiply %1357, %1360 : tensor<1x8x8x64xbf16>
    %1362 = stablehlo.subtract %1359, %1361 : tensor<1x8x8x64xbf16>
    %1363 = stablehlo.broadcast_in_dim %1353, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1364 = stablehlo.multiply %1357, %1363 : tensor<1x8x8x64xbf16>
    %1365 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1366 = stablehlo.multiply %1356, %1365 : tensor<1x8x8x64xbf16>
    %1367 = stablehlo.add %1364, %1366 : tensor<1x8x8x64xbf16>
    %1368 = stablehlo.concatenate %1362, %1367, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1369 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1370 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1371 = stablehlo.broadcast_in_dim %1369, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1372 = stablehlo.and %1371, %1370 : tensor<1x1x8x8xi1>
    %1373 = stablehlo.convert %1372 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1374 = sdy.sharding_constraint %1351 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1375 = sdy.sharding_constraint %1368 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1376 = sdy.sharding_constraint %1325 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1377 = sdy.sharding_constraint %1373 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1378 = stablehlo.reshape %1374 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1379 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1380 = stablehlo.multiply %1378, %1379 : tensor<1x8x8x2x128xbf16>
    %1381 = stablehlo.dot_general %1375, %1380, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1382 = stablehlo.transpose %1381, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1383 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1384 = stablehlo.compare  NE, %1377, %1383,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1385 = stablehlo.convert %1384 : tensor<1x1x8x8xi1>
    %1386 = stablehlo.broadcast_in_dim %1385, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1387 = stablehlo.reshape %1386 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1388 = call @_where_91(%1387, %1382, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1389 = stablehlo.convert %1388 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_63 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1390 = stablehlo.reduce(%1389 init: %cst_63) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1391 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1392 = stablehlo.maximum %1391, %1390 : tensor<1x8x2x8xf32>
    %1393 = stablehlo.broadcast_in_dim %1392, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1394 = stablehlo.broadcast_in_dim %1393, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1395 = stablehlo.subtract %1389, %1394 : tensor<1x8x2x8x8xf32>
    %1396 = stablehlo.exponential %1395 : tensor<1x8x2x8x8xf32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1397 = stablehlo.reduce(%1396 init: %cst_64) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1398 = stablehlo.broadcast_in_dim %1397, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1399 = stablehlo.broadcast_in_dim %1398, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1400 = stablehlo.divide %1396, %1399 : tensor<1x8x2x8x8xf32>
    %1401 = stablehlo.convert %1400 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1402 = stablehlo.dot_general %1376, %1401, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1403 = stablehlo.transpose %1402, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1404 = stablehlo.reshape %1403 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1405 = sdy.sharding_constraint %1404 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1406 = stablehlo.reshape %1405 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1407 = stablehlo.convert %arg197 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1408 = stablehlo.dot_general %1406, %1407, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1409 = stablehlo.add %1258, %1408 : tensor<1x8x1024xbf16>
    %1410 = stablehlo.convert %1409 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1411 = stablehlo.multiply %1410, %1410 : tensor<1x8x1024xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1412 = stablehlo.reduce(%1411 init: %cst_65) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1413 = stablehlo.broadcast_in_dim %1412, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1414 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1415 = stablehlo.divide %1413, %1414 : tensor<1x8x1xf32>
    %1416 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1417 = stablehlo.add %1415, %1416 : tensor<1x8x1xf32>
    %1418 = stablehlo.rsqrt %1417 : tensor<1x8x1xf32>
    %1419 = stablehlo.broadcast_in_dim %1418, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1420 = stablehlo.multiply %1410, %1419 : tensor<1x8x1024xf32>
    %1421 = stablehlo.convert %1420 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1422 = stablehlo.convert %arg194 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1424 = stablehlo.broadcast_in_dim %1423, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1425 = stablehlo.multiply %1424, %1421 : tensor<1x8x1024xbf16>
    %1426 = stablehlo.convert %arg192 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1427 = stablehlo.dot_general %1425, %1426, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1428 = call @silu(%1427) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1429 = stablehlo.convert %arg193 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1430 = stablehlo.dot_general %1425, %1429, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1431 = stablehlo.multiply %1428, %1430 : tensor<1x8x3072xbf16>
    %1432 = stablehlo.convert %arg191 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1433 = stablehlo.dot_general %1431, %1432, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1434 = stablehlo.add %1409, %1433 : tensor<1x8x1024xbf16>
    %1435 = stablehlo.convert %1434 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1436 = stablehlo.multiply %1435, %1435 : tensor<1x8x1024xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1437 = stablehlo.reduce(%1436 init: %cst_66) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1439 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1440 = stablehlo.divide %1438, %1439 : tensor<1x8x1xf32>
    %1441 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1442 = stablehlo.add %1440, %1441 : tensor<1x8x1xf32>
    %1443 = stablehlo.rsqrt %1442 : tensor<1x8x1xf32>
    %1444 = stablehlo.broadcast_in_dim %1443, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1445 = stablehlo.multiply %1435, %1444 : tensor<1x8x1024xf32>
    %1446 = stablehlo.convert %1445 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1447 = stablehlo.convert %arg201 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1448 = stablehlo.broadcast_in_dim %1447, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1449 = stablehlo.broadcast_in_dim %1448, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1450 = stablehlo.multiply %1449, %1446 : tensor<1x8x1024xbf16>
    %1451 = stablehlo.convert %arg32 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1452 = stablehlo.convert %arg33 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1453 = stablehlo.dot_general %1450, %1451, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1454 = stablehlo.dot_general %1453, %1452, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1455 = stablehlo.convert %arg210 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1456 = stablehlo.dot_general %1450, %1455, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1457 = stablehlo.add %1454, %1456 : tensor<1x8x2048xbf16>
    %1458 = stablehlo.convert %arg207 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1459 = stablehlo.dot_general %1450, %1458, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1460 = stablehlo.convert %arg34 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1461 = stablehlo.convert %arg35 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1462 = stablehlo.dot_general %1450, %1460, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1463 = stablehlo.dot_general %1462, %1461, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1464 = stablehlo.convert %arg211 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1465 = stablehlo.dot_general %1450, %1464, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1466 = stablehlo.add %1463, %1465 : tensor<1x8x1024xbf16>
    %1467 = stablehlo.reshape %1457 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1468 = stablehlo.convert %1467 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1469 = stablehlo.multiply %1468, %1468 : tensor<1x8x16x128xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1470 = stablehlo.reduce(%1469 init: %cst_67) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1471 = stablehlo.broadcast_in_dim %1470, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1472 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1473 = stablehlo.divide %1471, %1472 : tensor<1x8x16x1xf32>
    %1474 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1475 = stablehlo.add %1473, %1474 : tensor<1x8x16x1xf32>
    %1476 = stablehlo.rsqrt %1475 : tensor<1x8x16x1xf32>
    %1477 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1478 = stablehlo.multiply %1468, %1477 : tensor<1x8x16x128xf32>
    %1479 = stablehlo.convert %1478 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1480 = stablehlo.convert %arg209 : (tensor<128xf32>) -> tensor<128xbf16>
    %1481 = stablehlo.broadcast_in_dim %1480, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1482 = stablehlo.broadcast_in_dim %1481, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1483 = stablehlo.multiply %1482, %1479 : tensor<1x8x16x128xbf16>
    %1484 = stablehlo.reshape %1459 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1485 = stablehlo.convert %1484 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1486 = stablehlo.multiply %1485, %1485 : tensor<1x8x8x128xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1487 = stablehlo.reduce(%1486 init: %cst_68) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1488 = stablehlo.broadcast_in_dim %1487, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1489 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1490 = stablehlo.divide %1488, %1489 : tensor<1x8x8x1xf32>
    %1491 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1492 = stablehlo.add %1490, %1491 : tensor<1x8x8x1xf32>
    %1493 = stablehlo.rsqrt %1492 : tensor<1x8x8x1xf32>
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1495 = stablehlo.multiply %1485, %1494 : tensor<1x8x8x128xf32>
    %1496 = stablehlo.convert %1495 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1497 = stablehlo.convert %arg206 : (tensor<128xf32>) -> tensor<128xbf16>
    %1498 = stablehlo.broadcast_in_dim %1497, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1499 = stablehlo.broadcast_in_dim %1498, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1500 = stablehlo.multiply %1499, %1496 : tensor<1x8x8x128xbf16>
    %1501 = stablehlo.reshape %1466 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1502 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1503 = stablehlo.compare  LT, %7, %1502,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1504 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1505 = stablehlo.add %7, %1504 : tensor<1x8xi32>
    %1506 = stablehlo.select %1503, %1505, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1507 = stablehlo.broadcast_in_dim %1506, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1508 = "stablehlo.gather"(%26, %1507) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1509 = stablehlo.slice %1508 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1510 = stablehlo.slice %1508 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1511 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1512 = stablehlo.convert %1511 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1513 = stablehlo.broadcast_in_dim %1510, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1514 = stablehlo.convert %1513 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1515 = stablehlo.slice %1483 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1516 = stablehlo.slice %1483 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1517 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1518 = stablehlo.multiply %1515, %1517 : tensor<1x8x16x64xbf16>
    %1519 = stablehlo.broadcast_in_dim %1514, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1520 = stablehlo.multiply %1516, %1519 : tensor<1x8x16x64xbf16>
    %1521 = stablehlo.subtract %1518, %1520 : tensor<1x8x16x64xbf16>
    %1522 = stablehlo.broadcast_in_dim %1512, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1523 = stablehlo.multiply %1516, %1522 : tensor<1x8x16x64xbf16>
    %1524 = stablehlo.broadcast_in_dim %1514, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1525 = stablehlo.multiply %1515, %1524 : tensor<1x8x16x64xbf16>
    %1526 = stablehlo.add %1523, %1525 : tensor<1x8x16x64xbf16>
    %1527 = stablehlo.concatenate %1521, %1526, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1528 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1529 = stablehlo.convert %1528 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1530 = stablehlo.broadcast_in_dim %1510, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1531 = stablehlo.convert %1530 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1532 = stablehlo.slice %1500 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1533 = stablehlo.slice %1500 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1534 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1535 = stablehlo.multiply %1532, %1534 : tensor<1x8x8x64xbf16>
    %1536 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1537 = stablehlo.multiply %1533, %1536 : tensor<1x8x8x64xbf16>
    %1538 = stablehlo.subtract %1535, %1537 : tensor<1x8x8x64xbf16>
    %1539 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1540 = stablehlo.multiply %1533, %1539 : tensor<1x8x8x64xbf16>
    %1541 = stablehlo.broadcast_in_dim %1531, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1542 = stablehlo.multiply %1532, %1541 : tensor<1x8x8x64xbf16>
    %1543 = stablehlo.add %1540, %1542 : tensor<1x8x8x64xbf16>
    %1544 = stablehlo.concatenate %1538, %1543, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1545 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1546 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1547 = stablehlo.broadcast_in_dim %1545, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1548 = stablehlo.and %1547, %1546 : tensor<1x1x8x8xi1>
    %1549 = stablehlo.convert %1548 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1550 = sdy.sharding_constraint %1527 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1551 = sdy.sharding_constraint %1544 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1552 = sdy.sharding_constraint %1501 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1553 = sdy.sharding_constraint %1549 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1554 = stablehlo.reshape %1550 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1555 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1556 = stablehlo.multiply %1554, %1555 : tensor<1x8x8x2x128xbf16>
    %1557 = stablehlo.dot_general %1551, %1556, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1558 = stablehlo.transpose %1557, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1559 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1560 = stablehlo.compare  NE, %1553, %1559,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1561 = stablehlo.convert %1560 : tensor<1x1x8x8xi1>
    %1562 = stablehlo.broadcast_in_dim %1561, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1563 = stablehlo.reshape %1562 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1564 = call @_where_91(%1563, %1558, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1565 = stablehlo.convert %1564 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_70 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1566 = stablehlo.reduce(%1565 init: %cst_70) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1567 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1568 = stablehlo.maximum %1567, %1566 : tensor<1x8x2x8xf32>
    %1569 = stablehlo.broadcast_in_dim %1568, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1571 = stablehlo.subtract %1565, %1570 : tensor<1x8x2x8x8xf32>
    %1572 = stablehlo.exponential %1571 : tensor<1x8x2x8x8xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1573 = stablehlo.reduce(%1572 init: %cst_71) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1575 = stablehlo.broadcast_in_dim %1574, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1576 = stablehlo.divide %1572, %1575 : tensor<1x8x2x8x8xf32>
    %1577 = stablehlo.convert %1576 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1578 = stablehlo.dot_general %1552, %1577, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1579 = stablehlo.transpose %1578, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1580 = stablehlo.reshape %1579 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1581 = sdy.sharding_constraint %1580 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1582 = stablehlo.reshape %1581 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1583 = stablehlo.convert %arg208 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1584 = stablehlo.dot_general %1582, %1583, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1585 = stablehlo.add %1434, %1584 : tensor<1x8x1024xbf16>
    %1586 = stablehlo.convert %1585 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1587 = stablehlo.multiply %1586, %1586 : tensor<1x8x1024xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1588 = stablehlo.reduce(%1587 init: %cst_72) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1589 = stablehlo.broadcast_in_dim %1588, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1590 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1591 = stablehlo.divide %1589, %1590 : tensor<1x8x1xf32>
    %1592 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1593 = stablehlo.add %1591, %1592 : tensor<1x8x1xf32>
    %1594 = stablehlo.rsqrt %1593 : tensor<1x8x1xf32>
    %1595 = stablehlo.broadcast_in_dim %1594, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1596 = stablehlo.multiply %1586, %1595 : tensor<1x8x1024xf32>
    %1597 = stablehlo.convert %1596 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1598 = stablehlo.convert %arg205 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1599 = stablehlo.broadcast_in_dim %1598, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1601 = stablehlo.multiply %1600, %1597 : tensor<1x8x1024xbf16>
    %1602 = stablehlo.convert %arg203 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1603 = stablehlo.dot_general %1601, %1602, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1604 = call @silu(%1603) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1605 = stablehlo.convert %arg204 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1606 = stablehlo.dot_general %1601, %1605, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1607 = stablehlo.multiply %1604, %1606 : tensor<1x8x3072xbf16>
    %1608 = stablehlo.convert %arg202 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1609 = stablehlo.dot_general %1607, %1608, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1610 = stablehlo.add %1585, %1609 : tensor<1x8x1024xbf16>
    %1611 = stablehlo.convert %1610 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1612 = stablehlo.multiply %1611, %1611 : tensor<1x8x1024xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1613 = stablehlo.reduce(%1612 init: %cst_73) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1614 = stablehlo.broadcast_in_dim %1613, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1615 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1616 = stablehlo.divide %1614, %1615 : tensor<1x8x1xf32>
    %1617 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1618 = stablehlo.add %1616, %1617 : tensor<1x8x1xf32>
    %1619 = stablehlo.rsqrt %1618 : tensor<1x8x1xf32>
    %1620 = stablehlo.broadcast_in_dim %1619, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1621 = stablehlo.multiply %1611, %1620 : tensor<1x8x1024xf32>
    %1622 = stablehlo.convert %1621 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1623 = stablehlo.convert %arg212 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1624 = stablehlo.broadcast_in_dim %1623, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1625 = stablehlo.broadcast_in_dim %1624, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1626 = stablehlo.multiply %1625, %1622 : tensor<1x8x1024xbf16>
    %1627 = stablehlo.convert %arg36 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1628 = stablehlo.convert %arg37 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1629 = stablehlo.dot_general %1626, %1627, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1630 = stablehlo.dot_general %1629, %1628, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1631 = stablehlo.convert %arg221 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1632 = stablehlo.dot_general %1626, %1631, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1633 = stablehlo.add %1630, %1632 : tensor<1x8x2048xbf16>
    %1634 = stablehlo.convert %arg218 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1635 = stablehlo.dot_general %1626, %1634, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1636 = stablehlo.convert %arg38 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1637 = stablehlo.convert %arg39 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1638 = stablehlo.dot_general %1626, %1636, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1639 = stablehlo.dot_general %1638, %1637, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1640 = stablehlo.convert %arg222 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1641 = stablehlo.dot_general %1626, %1640, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1642 = stablehlo.add %1639, %1641 : tensor<1x8x1024xbf16>
    %1643 = stablehlo.reshape %1633 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1644 = stablehlo.convert %1643 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1645 = stablehlo.multiply %1644, %1644 : tensor<1x8x16x128xf32>
    %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1646 = stablehlo.reduce(%1645 init: %cst_74) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1647 = stablehlo.broadcast_in_dim %1646, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1648 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1649 = stablehlo.divide %1647, %1648 : tensor<1x8x16x1xf32>
    %1650 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1651 = stablehlo.add %1649, %1650 : tensor<1x8x16x1xf32>
    %1652 = stablehlo.rsqrt %1651 : tensor<1x8x16x1xf32>
    %1653 = stablehlo.broadcast_in_dim %1652, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1654 = stablehlo.multiply %1644, %1653 : tensor<1x8x16x128xf32>
    %1655 = stablehlo.convert %1654 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1656 = stablehlo.convert %arg220 : (tensor<128xf32>) -> tensor<128xbf16>
    %1657 = stablehlo.broadcast_in_dim %1656, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1658 = stablehlo.broadcast_in_dim %1657, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1659 = stablehlo.multiply %1658, %1655 : tensor<1x8x16x128xbf16>
    %1660 = stablehlo.reshape %1635 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1661 = stablehlo.convert %1660 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1662 = stablehlo.multiply %1661, %1661 : tensor<1x8x8x128xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1663 = stablehlo.reduce(%1662 init: %cst_75) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1664 = stablehlo.broadcast_in_dim %1663, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1665 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1666 = stablehlo.divide %1664, %1665 : tensor<1x8x8x1xf32>
    %1667 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1668 = stablehlo.add %1666, %1667 : tensor<1x8x8x1xf32>
    %1669 = stablehlo.rsqrt %1668 : tensor<1x8x8x1xf32>
    %1670 = stablehlo.broadcast_in_dim %1669, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1671 = stablehlo.multiply %1661, %1670 : tensor<1x8x8x128xf32>
    %1672 = stablehlo.convert %1671 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1673 = stablehlo.convert %arg217 : (tensor<128xf32>) -> tensor<128xbf16>
    %1674 = stablehlo.broadcast_in_dim %1673, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1676 = stablehlo.multiply %1675, %1672 : tensor<1x8x8x128xbf16>
    %1677 = stablehlo.reshape %1642 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1678 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1679 = stablehlo.compare  LT, %7, %1678,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1680 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1681 = stablehlo.add %7, %1680 : tensor<1x8xi32>
    %1682 = stablehlo.select %1679, %1681, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1683 = stablehlo.broadcast_in_dim %1682, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1684 = "stablehlo.gather"(%26, %1683) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1685 = stablehlo.slice %1684 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1686 = stablehlo.slice %1684 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1687 = stablehlo.broadcast_in_dim %1685, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1688 = stablehlo.convert %1687 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1689 = stablehlo.broadcast_in_dim %1686, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1690 = stablehlo.convert %1689 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1691 = stablehlo.slice %1659 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1692 = stablehlo.slice %1659 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1693 = stablehlo.broadcast_in_dim %1688, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1694 = stablehlo.multiply %1691, %1693 : tensor<1x8x16x64xbf16>
    %1695 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1696 = stablehlo.multiply %1692, %1695 : tensor<1x8x16x64xbf16>
    %1697 = stablehlo.subtract %1694, %1696 : tensor<1x8x16x64xbf16>
    %1698 = stablehlo.broadcast_in_dim %1688, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1699 = stablehlo.multiply %1692, %1698 : tensor<1x8x16x64xbf16>
    %1700 = stablehlo.broadcast_in_dim %1690, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1701 = stablehlo.multiply %1691, %1700 : tensor<1x8x16x64xbf16>
    %1702 = stablehlo.add %1699, %1701 : tensor<1x8x16x64xbf16>
    %1703 = stablehlo.concatenate %1697, %1702, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1704 = stablehlo.broadcast_in_dim %1685, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1705 = stablehlo.convert %1704 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1706 = stablehlo.broadcast_in_dim %1686, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1707 = stablehlo.convert %1706 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1708 = stablehlo.slice %1676 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1709 = stablehlo.slice %1676 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1710 = stablehlo.broadcast_in_dim %1705, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1711 = stablehlo.multiply %1708, %1710 : tensor<1x8x8x64xbf16>
    %1712 = stablehlo.broadcast_in_dim %1707, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1713 = stablehlo.multiply %1709, %1712 : tensor<1x8x8x64xbf16>
    %1714 = stablehlo.subtract %1711, %1713 : tensor<1x8x8x64xbf16>
    %1715 = stablehlo.broadcast_in_dim %1705, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1716 = stablehlo.multiply %1709, %1715 : tensor<1x8x8x64xbf16>
    %1717 = stablehlo.broadcast_in_dim %1707, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1718 = stablehlo.multiply %1708, %1717 : tensor<1x8x8x64xbf16>
    %1719 = stablehlo.add %1716, %1718 : tensor<1x8x8x64xbf16>
    %1720 = stablehlo.concatenate %1714, %1719, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1721 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1722 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1723 = stablehlo.broadcast_in_dim %1721, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1724 = stablehlo.and %1723, %1722 : tensor<1x1x8x8xi1>
    %1725 = stablehlo.convert %1724 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1726 = sdy.sharding_constraint %1703 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1727 = sdy.sharding_constraint %1720 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1728 = sdy.sharding_constraint %1677 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1729 = sdy.sharding_constraint %1725 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1730 = stablehlo.reshape %1726 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1731 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1732 = stablehlo.multiply %1730, %1731 : tensor<1x8x8x2x128xbf16>
    %1733 = stablehlo.dot_general %1727, %1732, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1734 = stablehlo.transpose %1733, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1735 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1736 = stablehlo.compare  NE, %1729, %1735,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1737 = stablehlo.convert %1736 : tensor<1x1x8x8xi1>
    %1738 = stablehlo.broadcast_in_dim %1737, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1739 = stablehlo.reshape %1738 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1740 = call @_where_91(%1739, %1734, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1741 = stablehlo.convert %1740 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_77 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1742 = stablehlo.reduce(%1741 init: %cst_77) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1743 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1744 = stablehlo.maximum %1743, %1742 : tensor<1x8x2x8xf32>
    %1745 = stablehlo.broadcast_in_dim %1744, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1747 = stablehlo.subtract %1741, %1746 : tensor<1x8x2x8x8xf32>
    %1748 = stablehlo.exponential %1747 : tensor<1x8x2x8x8xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1749 = stablehlo.reduce(%1748 init: %cst_78) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1750 = stablehlo.broadcast_in_dim %1749, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1752 = stablehlo.divide %1748, %1751 : tensor<1x8x2x8x8xf32>
    %1753 = stablehlo.convert %1752 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1754 = stablehlo.dot_general %1728, %1753, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1755 = stablehlo.transpose %1754, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1756 = stablehlo.reshape %1755 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1757 = sdy.sharding_constraint %1756 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1758 = stablehlo.reshape %1757 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1759 = stablehlo.convert %arg219 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1760 = stablehlo.dot_general %1758, %1759, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1761 = stablehlo.add %1610, %1760 : tensor<1x8x1024xbf16>
    %1762 = stablehlo.convert %1761 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1763 = stablehlo.multiply %1762, %1762 : tensor<1x8x1024xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1764 = stablehlo.reduce(%1763 init: %cst_79) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1765 = stablehlo.broadcast_in_dim %1764, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1766 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1767 = stablehlo.divide %1765, %1766 : tensor<1x8x1xf32>
    %1768 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1769 = stablehlo.add %1767, %1768 : tensor<1x8x1xf32>
    %1770 = stablehlo.rsqrt %1769 : tensor<1x8x1xf32>
    %1771 = stablehlo.broadcast_in_dim %1770, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1772 = stablehlo.multiply %1762, %1771 : tensor<1x8x1024xf32>
    %1773 = stablehlo.convert %1772 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1774 = stablehlo.convert %arg216 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1775 = stablehlo.broadcast_in_dim %1774, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1776 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1777 = stablehlo.multiply %1776, %1773 : tensor<1x8x1024xbf16>
    %1778 = stablehlo.convert %arg214 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1779 = stablehlo.dot_general %1777, %1778, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1780 = call @silu(%1779) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1781 = stablehlo.convert %arg215 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1782 = stablehlo.dot_general %1777, %1781, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1783 = stablehlo.multiply %1780, %1782 : tensor<1x8x3072xbf16>
    %1784 = stablehlo.convert %arg213 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1785 = stablehlo.dot_general %1783, %1784, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1786 = stablehlo.add %1761, %1785 : tensor<1x8x1024xbf16>
    %1787 = stablehlo.convert %1786 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1788 = stablehlo.multiply %1787, %1787 : tensor<1x8x1024xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1789 = stablehlo.reduce(%1788 init: %cst_80) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1790 = stablehlo.broadcast_in_dim %1789, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1791 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1792 = stablehlo.divide %1790, %1791 : tensor<1x8x1xf32>
    %1793 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1794 = stablehlo.add %1792, %1793 : tensor<1x8x1xf32>
    %1795 = stablehlo.rsqrt %1794 : tensor<1x8x1xf32>
    %1796 = stablehlo.broadcast_in_dim %1795, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1797 = stablehlo.multiply %1787, %1796 : tensor<1x8x1024xf32>
    %1798 = stablehlo.convert %1797 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1799 = stablehlo.convert %arg223 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1800 = stablehlo.broadcast_in_dim %1799, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1801 = stablehlo.broadcast_in_dim %1800, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1802 = stablehlo.multiply %1801, %1798 : tensor<1x8x1024xbf16>
    %1803 = stablehlo.convert %arg40 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1804 = stablehlo.convert %arg41 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1805 = stablehlo.dot_general %1802, %1803, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1806 = stablehlo.dot_general %1805, %1804, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1807 = stablehlo.convert %arg232 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1808 = stablehlo.dot_general %1802, %1807, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1809 = stablehlo.add %1806, %1808 : tensor<1x8x2048xbf16>
    %1810 = stablehlo.convert %arg229 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1811 = stablehlo.dot_general %1802, %1810, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1812 = stablehlo.convert %arg42 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1813 = stablehlo.convert %arg43 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1814 = stablehlo.dot_general %1802, %1812, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1815 = stablehlo.dot_general %1814, %1813, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1816 = stablehlo.convert %arg233 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1817 = stablehlo.dot_general %1802, %1816, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1818 = stablehlo.add %1815, %1817 : tensor<1x8x1024xbf16>
    %1819 = stablehlo.reshape %1809 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1820 = stablehlo.convert %1819 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1821 = stablehlo.multiply %1820, %1820 : tensor<1x8x16x128xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1822 = stablehlo.reduce(%1821 init: %cst_81) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1823 = stablehlo.broadcast_in_dim %1822, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1824 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1825 = stablehlo.divide %1823, %1824 : tensor<1x8x16x1xf32>
    %1826 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1827 = stablehlo.add %1825, %1826 : tensor<1x8x16x1xf32>
    %1828 = stablehlo.rsqrt %1827 : tensor<1x8x16x1xf32>
    %1829 = stablehlo.broadcast_in_dim %1828, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1830 = stablehlo.multiply %1820, %1829 : tensor<1x8x16x128xf32>
    %1831 = stablehlo.convert %1830 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1832 = stablehlo.convert %arg231 : (tensor<128xf32>) -> tensor<128xbf16>
    %1833 = stablehlo.broadcast_in_dim %1832, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1834 = stablehlo.broadcast_in_dim %1833, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1835 = stablehlo.multiply %1834, %1831 : tensor<1x8x16x128xbf16>
    %1836 = stablehlo.reshape %1811 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1837 = stablehlo.convert %1836 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1838 = stablehlo.multiply %1837, %1837 : tensor<1x8x8x128xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1839 = stablehlo.reduce(%1838 init: %cst_82) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1840 = stablehlo.broadcast_in_dim %1839, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1841 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1842 = stablehlo.divide %1840, %1841 : tensor<1x8x8x1xf32>
    %1843 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1844 = stablehlo.add %1842, %1843 : tensor<1x8x8x1xf32>
    %1845 = stablehlo.rsqrt %1844 : tensor<1x8x8x1xf32>
    %1846 = stablehlo.broadcast_in_dim %1845, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1847 = stablehlo.multiply %1837, %1846 : tensor<1x8x8x128xf32>
    %1848 = stablehlo.convert %1847 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1849 = stablehlo.convert %arg228 : (tensor<128xf32>) -> tensor<128xbf16>
    %1850 = stablehlo.broadcast_in_dim %1849, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1852 = stablehlo.multiply %1851, %1848 : tensor<1x8x8x128xbf16>
    %1853 = stablehlo.reshape %1818 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1854 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1855 = stablehlo.compare  LT, %7, %1854,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1856 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1857 = stablehlo.add %7, %1856 : tensor<1x8xi32>
    %1858 = stablehlo.select %1855, %1857, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1859 = stablehlo.broadcast_in_dim %1858, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1860 = "stablehlo.gather"(%26, %1859) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1861 = stablehlo.slice %1860 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1862 = stablehlo.slice %1860 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1863 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1864 = stablehlo.convert %1863 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1865 = stablehlo.broadcast_in_dim %1862, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1866 = stablehlo.convert %1865 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1867 = stablehlo.slice %1835 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1868 = stablehlo.slice %1835 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1869 = stablehlo.broadcast_in_dim %1864, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1870 = stablehlo.multiply %1867, %1869 : tensor<1x8x16x64xbf16>
    %1871 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1872 = stablehlo.multiply %1868, %1871 : tensor<1x8x16x64xbf16>
    %1873 = stablehlo.subtract %1870, %1872 : tensor<1x8x16x64xbf16>
    %1874 = stablehlo.broadcast_in_dim %1864, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1875 = stablehlo.multiply %1868, %1874 : tensor<1x8x16x64xbf16>
    %1876 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1877 = stablehlo.multiply %1867, %1876 : tensor<1x8x16x64xbf16>
    %1878 = stablehlo.add %1875, %1877 : tensor<1x8x16x64xbf16>
    %1879 = stablehlo.concatenate %1873, %1878, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1880 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1881 = stablehlo.convert %1880 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1882 = stablehlo.broadcast_in_dim %1862, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1883 = stablehlo.convert %1882 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1884 = stablehlo.slice %1852 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1885 = stablehlo.slice %1852 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1886 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1887 = stablehlo.multiply %1884, %1886 : tensor<1x8x8x64xbf16>
    %1888 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1889 = stablehlo.multiply %1885, %1888 : tensor<1x8x8x64xbf16>
    %1890 = stablehlo.subtract %1887, %1889 : tensor<1x8x8x64xbf16>
    %1891 = stablehlo.broadcast_in_dim %1881, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1892 = stablehlo.multiply %1885, %1891 : tensor<1x8x8x64xbf16>
    %1893 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1894 = stablehlo.multiply %1884, %1893 : tensor<1x8x8x64xbf16>
    %1895 = stablehlo.add %1892, %1894 : tensor<1x8x8x64xbf16>
    %1896 = stablehlo.concatenate %1890, %1895, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1897 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1898 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1899 = stablehlo.broadcast_in_dim %1897, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1900 = stablehlo.and %1899, %1898 : tensor<1x1x8x8xi1>
    %1901 = stablehlo.convert %1900 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1902 = sdy.sharding_constraint %1879 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1903 = sdy.sharding_constraint %1896 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1904 = sdy.sharding_constraint %1853 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1905 = sdy.sharding_constraint %1901 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1906 = stablehlo.reshape %1902 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1907 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1908 = stablehlo.multiply %1906, %1907 : tensor<1x8x8x2x128xbf16>
    %1909 = stablehlo.dot_general %1903, %1908, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1910 = stablehlo.transpose %1909, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1911 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1912 = stablehlo.compare  NE, %1905, %1911,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1913 = stablehlo.convert %1912 : tensor<1x1x8x8xi1>
    %1914 = stablehlo.broadcast_in_dim %1913, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1915 = stablehlo.reshape %1914 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1916 = call @_where_91(%1915, %1910, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1917 = stablehlo.convert %1916 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_84 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1918 = stablehlo.reduce(%1917 init: %cst_84) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1919 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1920 = stablehlo.maximum %1919, %1918 : tensor<1x8x2x8xf32>
    %1921 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1922 = stablehlo.broadcast_in_dim %1921, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1923 = stablehlo.subtract %1917, %1922 : tensor<1x8x2x8x8xf32>
    %1924 = stablehlo.exponential %1923 : tensor<1x8x2x8x8xf32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1925 = stablehlo.reduce(%1924 init: %cst_85) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1926 = stablehlo.broadcast_in_dim %1925, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1927 = stablehlo.broadcast_in_dim %1926, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1928 = stablehlo.divide %1924, %1927 : tensor<1x8x2x8x8xf32>
    %1929 = stablehlo.convert %1928 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1930 = stablehlo.dot_general %1904, %1929, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1931 = stablehlo.transpose %1930, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1932 = stablehlo.reshape %1931 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1933 = sdy.sharding_constraint %1932 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1934 = stablehlo.reshape %1933 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1935 = stablehlo.convert %arg230 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1936 = stablehlo.dot_general %1934, %1935, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1937 = stablehlo.add %1786, %1936 : tensor<1x8x1024xbf16>
    %1938 = stablehlo.convert %1937 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1939 = stablehlo.multiply %1938, %1938 : tensor<1x8x1024xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1940 = stablehlo.reduce(%1939 init: %cst_86) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1941 = stablehlo.broadcast_in_dim %1940, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1942 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1943 = stablehlo.divide %1941, %1942 : tensor<1x8x1xf32>
    %1944 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1945 = stablehlo.add %1943, %1944 : tensor<1x8x1xf32>
    %1946 = stablehlo.rsqrt %1945 : tensor<1x8x1xf32>
    %1947 = stablehlo.broadcast_in_dim %1946, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1948 = stablehlo.multiply %1938, %1947 : tensor<1x8x1024xf32>
    %1949 = stablehlo.convert %1948 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1950 = stablehlo.convert %arg227 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1951 = stablehlo.broadcast_in_dim %1950, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1952 = stablehlo.broadcast_in_dim %1951, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1953 = stablehlo.multiply %1952, %1949 : tensor<1x8x1024xbf16>
    %1954 = stablehlo.convert %arg225 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1955 = stablehlo.dot_general %1953, %1954, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1956 = call @silu(%1955) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1957 = stablehlo.convert %arg226 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1958 = stablehlo.dot_general %1953, %1957, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1959 = stablehlo.multiply %1956, %1958 : tensor<1x8x3072xbf16>
    %1960 = stablehlo.convert %arg224 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1961 = stablehlo.dot_general %1959, %1960, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1962 = stablehlo.add %1937, %1961 : tensor<1x8x1024xbf16>
    %1963 = stablehlo.convert %1962 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1964 = stablehlo.multiply %1963, %1963 : tensor<1x8x1024xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1965 = stablehlo.reduce(%1964 init: %cst_87) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1966 = stablehlo.broadcast_in_dim %1965, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1967 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1968 = stablehlo.divide %1966, %1967 : tensor<1x8x1xf32>
    %1969 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1970 = stablehlo.add %1968, %1969 : tensor<1x8x1xf32>
    %1971 = stablehlo.rsqrt %1970 : tensor<1x8x1xf32>
    %1972 = stablehlo.broadcast_in_dim %1971, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1973 = stablehlo.multiply %1963, %1972 : tensor<1x8x1024xf32>
    %1974 = stablehlo.convert %1973 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1975 = stablehlo.convert %arg234 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1976 = stablehlo.broadcast_in_dim %1975, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1977 = stablehlo.broadcast_in_dim %1976, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1978 = stablehlo.multiply %1977, %1974 : tensor<1x8x1024xbf16>
    %1979 = stablehlo.convert %arg44 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1980 = stablehlo.convert %arg45 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %1981 = stablehlo.dot_general %1978, %1979, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1982 = stablehlo.dot_general %1981, %1980, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1983 = stablehlo.convert %arg243 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1984 = stablehlo.dot_general %1978, %1983, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1985 = stablehlo.add %1982, %1984 : tensor<1x8x2048xbf16>
    %1986 = stablehlo.convert %arg240 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1987 = stablehlo.dot_general %1978, %1986, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1988 = stablehlo.convert %arg46 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %1989 = stablehlo.convert %arg47 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %1990 = stablehlo.dot_general %1978, %1988, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %1991 = stablehlo.dot_general %1990, %1989, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1992 = stablehlo.convert %arg244 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1993 = stablehlo.dot_general %1978, %1992, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1994 = stablehlo.add %1991, %1993 : tensor<1x8x1024xbf16>
    %1995 = stablehlo.reshape %1985 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1996 = stablehlo.convert %1995 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1997 = stablehlo.multiply %1996, %1996 : tensor<1x8x16x128xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1998 = stablehlo.reduce(%1997 init: %cst_88) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1999 = stablehlo.broadcast_in_dim %1998, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2000 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2001 = stablehlo.divide %1999, %2000 : tensor<1x8x16x1xf32>
    %2002 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2003 = stablehlo.add %2001, %2002 : tensor<1x8x16x1xf32>
    %2004 = stablehlo.rsqrt %2003 : tensor<1x8x16x1xf32>
    %2005 = stablehlo.broadcast_in_dim %2004, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2006 = stablehlo.multiply %1996, %2005 : tensor<1x8x16x128xf32>
    %2007 = stablehlo.convert %2006 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2008 = stablehlo.convert %arg242 : (tensor<128xf32>) -> tensor<128xbf16>
    %2009 = stablehlo.broadcast_in_dim %2008, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2010 = stablehlo.broadcast_in_dim %2009, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2011 = stablehlo.multiply %2010, %2007 : tensor<1x8x16x128xbf16>
    %2012 = stablehlo.reshape %1987 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2013 = stablehlo.convert %2012 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2014 = stablehlo.multiply %2013, %2013 : tensor<1x8x8x128xf32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2015 = stablehlo.reduce(%2014 init: %cst_89) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2016 = stablehlo.broadcast_in_dim %2015, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2017 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2018 = stablehlo.divide %2016, %2017 : tensor<1x8x8x1xf32>
    %2019 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2020 = stablehlo.add %2018, %2019 : tensor<1x8x8x1xf32>
    %2021 = stablehlo.rsqrt %2020 : tensor<1x8x8x1xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2023 = stablehlo.multiply %2013, %2022 : tensor<1x8x8x128xf32>
    %2024 = stablehlo.convert %2023 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2025 = stablehlo.convert %arg239 : (tensor<128xf32>) -> tensor<128xbf16>
    %2026 = stablehlo.broadcast_in_dim %2025, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2027 = stablehlo.broadcast_in_dim %2026, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2028 = stablehlo.multiply %2027, %2024 : tensor<1x8x8x128xbf16>
    %2029 = stablehlo.reshape %1994 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2030 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2031 = stablehlo.compare  LT, %7, %2030,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2032 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2033 = stablehlo.add %7, %2032 : tensor<1x8xi32>
    %2034 = stablehlo.select %2031, %2033, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2035 = stablehlo.broadcast_in_dim %2034, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2036 = "stablehlo.gather"(%26, %2035) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2037 = stablehlo.slice %2036 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2038 = stablehlo.slice %2036 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2039 = stablehlo.broadcast_in_dim %2037, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2040 = stablehlo.convert %2039 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2041 = stablehlo.broadcast_in_dim %2038, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2042 = stablehlo.convert %2041 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2043 = stablehlo.slice %2011 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2044 = stablehlo.slice %2011 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2045 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2046 = stablehlo.multiply %2043, %2045 : tensor<1x8x16x64xbf16>
    %2047 = stablehlo.broadcast_in_dim %2042, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2048 = stablehlo.multiply %2044, %2047 : tensor<1x8x16x64xbf16>
    %2049 = stablehlo.subtract %2046, %2048 : tensor<1x8x16x64xbf16>
    %2050 = stablehlo.broadcast_in_dim %2040, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2051 = stablehlo.multiply %2044, %2050 : tensor<1x8x16x64xbf16>
    %2052 = stablehlo.broadcast_in_dim %2042, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2053 = stablehlo.multiply %2043, %2052 : tensor<1x8x16x64xbf16>
    %2054 = stablehlo.add %2051, %2053 : tensor<1x8x16x64xbf16>
    %2055 = stablehlo.concatenate %2049, %2054, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2056 = stablehlo.broadcast_in_dim %2037, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2057 = stablehlo.convert %2056 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2058 = stablehlo.broadcast_in_dim %2038, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2059 = stablehlo.convert %2058 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2060 = stablehlo.slice %2028 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2061 = stablehlo.slice %2028 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2062 = stablehlo.broadcast_in_dim %2057, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2063 = stablehlo.multiply %2060, %2062 : tensor<1x8x8x64xbf16>
    %2064 = stablehlo.broadcast_in_dim %2059, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2065 = stablehlo.multiply %2061, %2064 : tensor<1x8x8x64xbf16>
    %2066 = stablehlo.subtract %2063, %2065 : tensor<1x8x8x64xbf16>
    %2067 = stablehlo.broadcast_in_dim %2057, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2068 = stablehlo.multiply %2061, %2067 : tensor<1x8x8x64xbf16>
    %2069 = stablehlo.broadcast_in_dim %2059, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2070 = stablehlo.multiply %2060, %2069 : tensor<1x8x8x64xbf16>
    %2071 = stablehlo.add %2068, %2070 : tensor<1x8x8x64xbf16>
    %2072 = stablehlo.concatenate %2066, %2071, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2073 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2074 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2075 = stablehlo.broadcast_in_dim %2073, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2076 = stablehlo.and %2075, %2074 : tensor<1x1x8x8xi1>
    %2077 = stablehlo.convert %2076 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2078 = sdy.sharding_constraint %2055 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2079 = sdy.sharding_constraint %2072 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2080 = sdy.sharding_constraint %2029 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2081 = sdy.sharding_constraint %2077 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2082 = stablehlo.reshape %2078 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2083 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2084 = stablehlo.multiply %2082, %2083 : tensor<1x8x8x2x128xbf16>
    %2085 = stablehlo.dot_general %2079, %2084, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2086 = stablehlo.transpose %2085, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2087 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2088 = stablehlo.compare  NE, %2081, %2087,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2089 = stablehlo.convert %2088 : tensor<1x1x8x8xi1>
    %2090 = stablehlo.broadcast_in_dim %2089, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2091 = stablehlo.reshape %2090 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2092 = call @_where_91(%2091, %2086, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2093 = stablehlo.convert %2092 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_91 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2094 = stablehlo.reduce(%2093 init: %cst_91) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2095 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2096 = stablehlo.maximum %2095, %2094 : tensor<1x8x2x8xf32>
    %2097 = stablehlo.broadcast_in_dim %2096, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2098 = stablehlo.broadcast_in_dim %2097, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2099 = stablehlo.subtract %2093, %2098 : tensor<1x8x2x8x8xf32>
    %2100 = stablehlo.exponential %2099 : tensor<1x8x2x8x8xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2101 = stablehlo.reduce(%2100 init: %cst_92) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2104 = stablehlo.divide %2100, %2103 : tensor<1x8x2x8x8xf32>
    %2105 = stablehlo.convert %2104 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2106 = stablehlo.dot_general %2080, %2105, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2107 = stablehlo.transpose %2106, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2108 = stablehlo.reshape %2107 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2109 = sdy.sharding_constraint %2108 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2110 = stablehlo.reshape %2109 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2111 = stablehlo.convert %arg241 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2112 = stablehlo.dot_general %2110, %2111, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2113 = stablehlo.add %1962, %2112 : tensor<1x8x1024xbf16>
    %2114 = stablehlo.convert %2113 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2115 = stablehlo.multiply %2114, %2114 : tensor<1x8x1024xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2116 = stablehlo.reduce(%2115 init: %cst_93) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2117 = stablehlo.broadcast_in_dim %2116, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2118 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2119 = stablehlo.divide %2117, %2118 : tensor<1x8x1xf32>
    %2120 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2121 = stablehlo.add %2119, %2120 : tensor<1x8x1xf32>
    %2122 = stablehlo.rsqrt %2121 : tensor<1x8x1xf32>
    %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2124 = stablehlo.multiply %2114, %2123 : tensor<1x8x1024xf32>
    %2125 = stablehlo.convert %2124 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2126 = stablehlo.convert %arg238 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2127 = stablehlo.broadcast_in_dim %2126, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2128 = stablehlo.broadcast_in_dim %2127, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2129 = stablehlo.multiply %2128, %2125 : tensor<1x8x1024xbf16>
    %2130 = stablehlo.convert %arg236 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2131 = stablehlo.dot_general %2129, %2130, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2132 = call @silu(%2131) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2133 = stablehlo.convert %arg237 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2134 = stablehlo.dot_general %2129, %2133, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2135 = stablehlo.multiply %2132, %2134 : tensor<1x8x3072xbf16>
    %2136 = stablehlo.convert %arg235 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2137 = stablehlo.dot_general %2135, %2136, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2138 = stablehlo.add %2113, %2137 : tensor<1x8x1024xbf16>
    %2139 = stablehlo.convert %2138 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2140 = stablehlo.multiply %2139, %2139 : tensor<1x8x1024xf32>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2141 = stablehlo.reduce(%2140 init: %cst_94) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2142 = stablehlo.broadcast_in_dim %2141, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2143 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2144 = stablehlo.divide %2142, %2143 : tensor<1x8x1xf32>
    %2145 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2146 = stablehlo.add %2144, %2145 : tensor<1x8x1xf32>
    %2147 = stablehlo.rsqrt %2146 : tensor<1x8x1xf32>
    %2148 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2149 = stablehlo.multiply %2139, %2148 : tensor<1x8x1024xf32>
    %2150 = stablehlo.convert %2149 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2151 = stablehlo.convert %arg245 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2152 = stablehlo.broadcast_in_dim %2151, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2154 = stablehlo.multiply %2153, %2150 : tensor<1x8x1024xbf16>
    %2155 = stablehlo.convert %arg48 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2156 = stablehlo.convert %arg49 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %2157 = stablehlo.dot_general %2154, %2155, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2158 = stablehlo.dot_general %2157, %2156, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2159 = stablehlo.convert %arg254 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2160 = stablehlo.dot_general %2154, %2159, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2161 = stablehlo.add %2158, %2160 : tensor<1x8x2048xbf16>
    %2162 = stablehlo.convert %arg251 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2163 = stablehlo.dot_general %2154, %2162, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2164 = stablehlo.convert %arg50 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2165 = stablehlo.convert %arg51 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %2166 = stablehlo.dot_general %2154, %2164, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2167 = stablehlo.dot_general %2166, %2165, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2168 = stablehlo.convert %arg255 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2169 = stablehlo.dot_general %2154, %2168, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2170 = stablehlo.add %2167, %2169 : tensor<1x8x1024xbf16>
    %2171 = stablehlo.reshape %2161 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2172 = stablehlo.convert %2171 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2173 = stablehlo.multiply %2172, %2172 : tensor<1x8x16x128xf32>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2174 = stablehlo.reduce(%2173 init: %cst_95) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2175 = stablehlo.broadcast_in_dim %2174, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2176 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2177 = stablehlo.divide %2175, %2176 : tensor<1x8x16x1xf32>
    %2178 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2179 = stablehlo.add %2177, %2178 : tensor<1x8x16x1xf32>
    %2180 = stablehlo.rsqrt %2179 : tensor<1x8x16x1xf32>
    %2181 = stablehlo.broadcast_in_dim %2180, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2182 = stablehlo.multiply %2172, %2181 : tensor<1x8x16x128xf32>
    %2183 = stablehlo.convert %2182 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2184 = stablehlo.convert %arg253 : (tensor<128xf32>) -> tensor<128xbf16>
    %2185 = stablehlo.broadcast_in_dim %2184, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2186 = stablehlo.broadcast_in_dim %2185, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2187 = stablehlo.multiply %2186, %2183 : tensor<1x8x16x128xbf16>
    %2188 = stablehlo.reshape %2163 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2189 = stablehlo.convert %2188 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2190 = stablehlo.multiply %2189, %2189 : tensor<1x8x8x128xf32>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2191 = stablehlo.reduce(%2190 init: %cst_96) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2192 = stablehlo.broadcast_in_dim %2191, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2193 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2194 = stablehlo.divide %2192, %2193 : tensor<1x8x8x1xf32>
    %2195 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2196 = stablehlo.add %2194, %2195 : tensor<1x8x8x1xf32>
    %2197 = stablehlo.rsqrt %2196 : tensor<1x8x8x1xf32>
    %2198 = stablehlo.broadcast_in_dim %2197, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2199 = stablehlo.multiply %2189, %2198 : tensor<1x8x8x128xf32>
    %2200 = stablehlo.convert %2199 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2201 = stablehlo.convert %arg250 : (tensor<128xf32>) -> tensor<128xbf16>
    %2202 = stablehlo.broadcast_in_dim %2201, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2203 = stablehlo.broadcast_in_dim %2202, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2204 = stablehlo.multiply %2203, %2200 : tensor<1x8x8x128xbf16>
    %2205 = stablehlo.reshape %2170 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2206 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2207 = stablehlo.compare  LT, %7, %2206,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2208 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2209 = stablehlo.add %7, %2208 : tensor<1x8xi32>
    %2210 = stablehlo.select %2207, %2209, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2211 = stablehlo.broadcast_in_dim %2210, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2212 = "stablehlo.gather"(%26, %2211) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2213 = stablehlo.slice %2212 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2214 = stablehlo.slice %2212 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2215 = stablehlo.broadcast_in_dim %2213, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2216 = stablehlo.convert %2215 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2217 = stablehlo.broadcast_in_dim %2214, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2218 = stablehlo.convert %2217 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2219 = stablehlo.slice %2187 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2220 = stablehlo.slice %2187 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2221 = stablehlo.broadcast_in_dim %2216, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2222 = stablehlo.multiply %2219, %2221 : tensor<1x8x16x64xbf16>
    %2223 = stablehlo.broadcast_in_dim %2218, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2224 = stablehlo.multiply %2220, %2223 : tensor<1x8x16x64xbf16>
    %2225 = stablehlo.subtract %2222, %2224 : tensor<1x8x16x64xbf16>
    %2226 = stablehlo.broadcast_in_dim %2216, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2227 = stablehlo.multiply %2220, %2226 : tensor<1x8x16x64xbf16>
    %2228 = stablehlo.broadcast_in_dim %2218, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2229 = stablehlo.multiply %2219, %2228 : tensor<1x8x16x64xbf16>
    %2230 = stablehlo.add %2227, %2229 : tensor<1x8x16x64xbf16>
    %2231 = stablehlo.concatenate %2225, %2230, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2232 = stablehlo.broadcast_in_dim %2213, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2233 = stablehlo.convert %2232 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2234 = stablehlo.broadcast_in_dim %2214, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2235 = stablehlo.convert %2234 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2236 = stablehlo.slice %2204 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2237 = stablehlo.slice %2204 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2238 = stablehlo.broadcast_in_dim %2233, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2239 = stablehlo.multiply %2236, %2238 : tensor<1x8x8x64xbf16>
    %2240 = stablehlo.broadcast_in_dim %2235, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2241 = stablehlo.multiply %2237, %2240 : tensor<1x8x8x64xbf16>
    %2242 = stablehlo.subtract %2239, %2241 : tensor<1x8x8x64xbf16>
    %2243 = stablehlo.broadcast_in_dim %2233, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2244 = stablehlo.multiply %2237, %2243 : tensor<1x8x8x64xbf16>
    %2245 = stablehlo.broadcast_in_dim %2235, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2246 = stablehlo.multiply %2236, %2245 : tensor<1x8x8x64xbf16>
    %2247 = stablehlo.add %2244, %2246 : tensor<1x8x8x64xbf16>
    %2248 = stablehlo.concatenate %2242, %2247, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2249 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2250 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2251 = stablehlo.broadcast_in_dim %2249, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2252 = stablehlo.and %2251, %2250 : tensor<1x1x8x8xi1>
    %2253 = stablehlo.convert %2252 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2254 = sdy.sharding_constraint %2231 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2255 = sdy.sharding_constraint %2248 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2256 = sdy.sharding_constraint %2205 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2257 = sdy.sharding_constraint %2253 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2258 = stablehlo.reshape %2254 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2259 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2260 = stablehlo.multiply %2258, %2259 : tensor<1x8x8x2x128xbf16>
    %2261 = stablehlo.dot_general %2255, %2260, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2262 = stablehlo.transpose %2261, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2263 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2264 = stablehlo.compare  NE, %2257, %2263,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2265 = stablehlo.convert %2264 : tensor<1x1x8x8xi1>
    %2266 = stablehlo.broadcast_in_dim %2265, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2267 = stablehlo.reshape %2266 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2268 = call @_where_91(%2267, %2262, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2269 = stablehlo.convert %2268 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_98 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2270 = stablehlo.reduce(%2269 init: %cst_98) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2271 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2272 = stablehlo.maximum %2271, %2270 : tensor<1x8x2x8xf32>
    %2273 = stablehlo.broadcast_in_dim %2272, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2274 = stablehlo.broadcast_in_dim %2273, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2275 = stablehlo.subtract %2269, %2274 : tensor<1x8x2x8x8xf32>
    %2276 = stablehlo.exponential %2275 : tensor<1x8x2x8x8xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2277 = stablehlo.reduce(%2276 init: %cst_99) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2278 = stablehlo.broadcast_in_dim %2277, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2279 = stablehlo.broadcast_in_dim %2278, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2280 = stablehlo.divide %2276, %2279 : tensor<1x8x2x8x8xf32>
    %2281 = stablehlo.convert %2280 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2282 = stablehlo.dot_general %2256, %2281, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2283 = stablehlo.transpose %2282, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2284 = stablehlo.reshape %2283 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2285 = sdy.sharding_constraint %2284 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2286 = stablehlo.reshape %2285 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2287 = stablehlo.convert %arg252 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2288 = stablehlo.dot_general %2286, %2287, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2289 = stablehlo.add %2138, %2288 : tensor<1x8x1024xbf16>
    %2290 = stablehlo.convert %2289 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2291 = stablehlo.multiply %2290, %2290 : tensor<1x8x1024xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2292 = stablehlo.reduce(%2291 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2293 = stablehlo.broadcast_in_dim %2292, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2294 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2295 = stablehlo.divide %2293, %2294 : tensor<1x8x1xf32>
    %2296 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2297 = stablehlo.add %2295, %2296 : tensor<1x8x1xf32>
    %2298 = stablehlo.rsqrt %2297 : tensor<1x8x1xf32>
    %2299 = stablehlo.broadcast_in_dim %2298, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2300 = stablehlo.multiply %2290, %2299 : tensor<1x8x1024xf32>
    %2301 = stablehlo.convert %2300 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2302 = stablehlo.convert %arg249 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2303 = stablehlo.broadcast_in_dim %2302, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2304 = stablehlo.broadcast_in_dim %2303, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2305 = stablehlo.multiply %2304, %2301 : tensor<1x8x1024xbf16>
    %2306 = stablehlo.convert %arg247 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2307 = stablehlo.dot_general %2305, %2306, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2308 = call @silu(%2307) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2309 = stablehlo.convert %arg248 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2310 = stablehlo.dot_general %2305, %2309, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2311 = stablehlo.multiply %2308, %2310 : tensor<1x8x3072xbf16>
    %2312 = stablehlo.convert %arg246 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2313 = stablehlo.dot_general %2311, %2312, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2314 = stablehlo.add %2289, %2313 : tensor<1x8x1024xbf16>
    %2315 = stablehlo.convert %2314 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2316 = stablehlo.multiply %2315, %2315 : tensor<1x8x1024xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2317 = stablehlo.reduce(%2316 init: %cst_101) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2318 = stablehlo.broadcast_in_dim %2317, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2319 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2320 = stablehlo.divide %2318, %2319 : tensor<1x8x1xf32>
    %2321 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2322 = stablehlo.add %2320, %2321 : tensor<1x8x1xf32>
    %2323 = stablehlo.rsqrt %2322 : tensor<1x8x1xf32>
    %2324 = stablehlo.broadcast_in_dim %2323, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2325 = stablehlo.multiply %2315, %2324 : tensor<1x8x1024xf32>
    %2326 = stablehlo.convert %2325 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2327 = stablehlo.convert %arg256 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2328 = stablehlo.broadcast_in_dim %2327, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2329 = stablehlo.broadcast_in_dim %2328, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2330 = stablehlo.multiply %2329, %2326 : tensor<1x8x1024xbf16>
    %2331 = stablehlo.convert %arg52 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2332 = stablehlo.convert %arg53 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %2333 = stablehlo.dot_general %2330, %2331, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2334 = stablehlo.dot_general %2333, %2332, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2335 = stablehlo.convert %arg265 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2336 = stablehlo.dot_general %2330, %2335, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2337 = stablehlo.add %2334, %2336 : tensor<1x8x2048xbf16>
    %2338 = stablehlo.convert %arg262 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2339 = stablehlo.dot_general %2330, %2338, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2340 = stablehlo.convert %arg54 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2341 = stablehlo.convert %arg55 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %2342 = stablehlo.dot_general %2330, %2340, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2343 = stablehlo.dot_general %2342, %2341, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2344 = stablehlo.convert %arg266 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2345 = stablehlo.dot_general %2330, %2344, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2346 = stablehlo.add %2343, %2345 : tensor<1x8x1024xbf16>
    %2347 = stablehlo.reshape %2337 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2348 = stablehlo.convert %2347 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2349 = stablehlo.multiply %2348, %2348 : tensor<1x8x16x128xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2350 = stablehlo.reduce(%2349 init: %cst_102) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2352 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2353 = stablehlo.divide %2351, %2352 : tensor<1x8x16x1xf32>
    %2354 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2355 = stablehlo.add %2353, %2354 : tensor<1x8x16x1xf32>
    %2356 = stablehlo.rsqrt %2355 : tensor<1x8x16x1xf32>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2358 = stablehlo.multiply %2348, %2357 : tensor<1x8x16x128xf32>
    %2359 = stablehlo.convert %2358 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2360 = stablehlo.convert %arg264 : (tensor<128xf32>) -> tensor<128xbf16>
    %2361 = stablehlo.broadcast_in_dim %2360, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2362 = stablehlo.broadcast_in_dim %2361, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2363 = stablehlo.multiply %2362, %2359 : tensor<1x8x16x128xbf16>
    %2364 = stablehlo.reshape %2339 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2365 = stablehlo.convert %2364 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2366 = stablehlo.multiply %2365, %2365 : tensor<1x8x8x128xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2367 = stablehlo.reduce(%2366 init: %cst_103) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2368 = stablehlo.broadcast_in_dim %2367, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2369 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2370 = stablehlo.divide %2368, %2369 : tensor<1x8x8x1xf32>
    %2371 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2372 = stablehlo.add %2370, %2371 : tensor<1x8x8x1xf32>
    %2373 = stablehlo.rsqrt %2372 : tensor<1x8x8x1xf32>
    %2374 = stablehlo.broadcast_in_dim %2373, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2375 = stablehlo.multiply %2365, %2374 : tensor<1x8x8x128xf32>
    %2376 = stablehlo.convert %2375 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2377 = stablehlo.convert %arg261 : (tensor<128xf32>) -> tensor<128xbf16>
    %2378 = stablehlo.broadcast_in_dim %2377, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2379 = stablehlo.broadcast_in_dim %2378, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2380 = stablehlo.multiply %2379, %2376 : tensor<1x8x8x128xbf16>
    %2381 = stablehlo.reshape %2346 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2382 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2383 = stablehlo.compare  LT, %7, %2382,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2384 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2385 = stablehlo.add %7, %2384 : tensor<1x8xi32>
    %2386 = stablehlo.select %2383, %2385, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2387 = stablehlo.broadcast_in_dim %2386, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2388 = "stablehlo.gather"(%26, %2387) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2389 = stablehlo.slice %2388 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2390 = stablehlo.slice %2388 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2391 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2392 = stablehlo.convert %2391 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2393 = stablehlo.broadcast_in_dim %2390, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2394 = stablehlo.convert %2393 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2395 = stablehlo.slice %2363 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2396 = stablehlo.slice %2363 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2397 = stablehlo.broadcast_in_dim %2392, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2398 = stablehlo.multiply %2395, %2397 : tensor<1x8x16x64xbf16>
    %2399 = stablehlo.broadcast_in_dim %2394, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2400 = stablehlo.multiply %2396, %2399 : tensor<1x8x16x64xbf16>
    %2401 = stablehlo.subtract %2398, %2400 : tensor<1x8x16x64xbf16>
    %2402 = stablehlo.broadcast_in_dim %2392, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2403 = stablehlo.multiply %2396, %2402 : tensor<1x8x16x64xbf16>
    %2404 = stablehlo.broadcast_in_dim %2394, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2405 = stablehlo.multiply %2395, %2404 : tensor<1x8x16x64xbf16>
    %2406 = stablehlo.add %2403, %2405 : tensor<1x8x16x64xbf16>
    %2407 = stablehlo.concatenate %2401, %2406, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2408 = stablehlo.broadcast_in_dim %2389, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2409 = stablehlo.convert %2408 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2410 = stablehlo.broadcast_in_dim %2390, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2411 = stablehlo.convert %2410 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2412 = stablehlo.slice %2380 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2413 = stablehlo.slice %2380 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2414 = stablehlo.broadcast_in_dim %2409, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2415 = stablehlo.multiply %2412, %2414 : tensor<1x8x8x64xbf16>
    %2416 = stablehlo.broadcast_in_dim %2411, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2417 = stablehlo.multiply %2413, %2416 : tensor<1x8x8x64xbf16>
    %2418 = stablehlo.subtract %2415, %2417 : tensor<1x8x8x64xbf16>
    %2419 = stablehlo.broadcast_in_dim %2409, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2420 = stablehlo.multiply %2413, %2419 : tensor<1x8x8x64xbf16>
    %2421 = stablehlo.broadcast_in_dim %2411, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2422 = stablehlo.multiply %2412, %2421 : tensor<1x8x8x64xbf16>
    %2423 = stablehlo.add %2420, %2422 : tensor<1x8x8x64xbf16>
    %2424 = stablehlo.concatenate %2418, %2423, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2425 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2426 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2427 = stablehlo.broadcast_in_dim %2425, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2428 = stablehlo.and %2427, %2426 : tensor<1x1x8x8xi1>
    %2429 = stablehlo.convert %2428 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2430 = sdy.sharding_constraint %2407 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2431 = sdy.sharding_constraint %2424 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2432 = sdy.sharding_constraint %2381 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2433 = sdy.sharding_constraint %2429 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2434 = stablehlo.reshape %2430 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2435 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2436 = stablehlo.multiply %2434, %2435 : tensor<1x8x8x2x128xbf16>
    %2437 = stablehlo.dot_general %2431, %2436, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2438 = stablehlo.transpose %2437, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2439 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2440 = stablehlo.compare  NE, %2433, %2439,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2441 = stablehlo.convert %2440 : tensor<1x1x8x8xi1>
    %2442 = stablehlo.broadcast_in_dim %2441, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2443 = stablehlo.reshape %2442 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2444 = call @_where_91(%2443, %2438, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2445 = stablehlo.convert %2444 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_105 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2446 = stablehlo.reduce(%2445 init: %cst_105) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2447 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2448 = stablehlo.maximum %2447, %2446 : tensor<1x8x2x8xf32>
    %2449 = stablehlo.broadcast_in_dim %2448, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2450 = stablehlo.broadcast_in_dim %2449, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2451 = stablehlo.subtract %2445, %2450 : tensor<1x8x2x8x8xf32>
    %2452 = stablehlo.exponential %2451 : tensor<1x8x2x8x8xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2453 = stablehlo.reduce(%2452 init: %cst_106) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2454 = stablehlo.broadcast_in_dim %2453, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2455 = stablehlo.broadcast_in_dim %2454, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2456 = stablehlo.divide %2452, %2455 : tensor<1x8x2x8x8xf32>
    %2457 = stablehlo.convert %2456 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2458 = stablehlo.dot_general %2432, %2457, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2459 = stablehlo.transpose %2458, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2460 = stablehlo.reshape %2459 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2461 = sdy.sharding_constraint %2460 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2462 = stablehlo.reshape %2461 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2463 = stablehlo.convert %arg263 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2464 = stablehlo.dot_general %2462, %2463, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2465 = stablehlo.add %2314, %2464 : tensor<1x8x1024xbf16>
    %2466 = stablehlo.convert %2465 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2467 = stablehlo.multiply %2466, %2466 : tensor<1x8x1024xf32>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2468 = stablehlo.reduce(%2467 init: %cst_107) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2469 = stablehlo.broadcast_in_dim %2468, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2470 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2471 = stablehlo.divide %2469, %2470 : tensor<1x8x1xf32>
    %2472 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2473 = stablehlo.add %2471, %2472 : tensor<1x8x1xf32>
    %2474 = stablehlo.rsqrt %2473 : tensor<1x8x1xf32>
    %2475 = stablehlo.broadcast_in_dim %2474, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2476 = stablehlo.multiply %2466, %2475 : tensor<1x8x1024xf32>
    %2477 = stablehlo.convert %2476 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2478 = stablehlo.convert %arg260 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2479 = stablehlo.broadcast_in_dim %2478, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2480 = stablehlo.broadcast_in_dim %2479, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2481 = stablehlo.multiply %2480, %2477 : tensor<1x8x1024xbf16>
    %2482 = stablehlo.convert %arg258 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2483 = stablehlo.dot_general %2481, %2482, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2484 = call @silu(%2483) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2485 = stablehlo.convert %arg259 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2486 = stablehlo.dot_general %2481, %2485, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2487 = stablehlo.multiply %2484, %2486 : tensor<1x8x3072xbf16>
    %2488 = stablehlo.convert %arg257 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2489 = stablehlo.dot_general %2487, %2488, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2490 = stablehlo.add %2465, %2489 : tensor<1x8x1024xbf16>
    %2491 = stablehlo.convert %2490 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2492 = stablehlo.multiply %2491, %2491 : tensor<1x8x1024xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2493 = stablehlo.reduce(%2492 init: %cst_108) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2494 = stablehlo.broadcast_in_dim %2493, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2495 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2496 = stablehlo.divide %2494, %2495 : tensor<1x8x1xf32>
    %2497 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2498 = stablehlo.add %2496, %2497 : tensor<1x8x1xf32>
    %2499 = stablehlo.rsqrt %2498 : tensor<1x8x1xf32>
    %2500 = stablehlo.broadcast_in_dim %2499, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2501 = stablehlo.multiply %2491, %2500 : tensor<1x8x1024xf32>
    %2502 = stablehlo.convert %2501 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2503 = stablehlo.convert %arg267 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2504 = stablehlo.broadcast_in_dim %2503, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2505 = stablehlo.broadcast_in_dim %2504, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2506 = stablehlo.multiply %2505, %2502 : tensor<1x8x1024xbf16>
    %2507 = stablehlo.convert %arg56 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2508 = stablehlo.convert %arg57 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %2509 = stablehlo.dot_general %2506, %2507, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2510 = stablehlo.dot_general %2509, %2508, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2511 = stablehlo.convert %arg276 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2512 = stablehlo.dot_general %2506, %2511, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2513 = stablehlo.add %2510, %2512 : tensor<1x8x2048xbf16>
    %2514 = stablehlo.convert %arg273 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2515 = stablehlo.dot_general %2506, %2514, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2516 = stablehlo.convert %arg58 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2517 = stablehlo.convert %arg59 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %2518 = stablehlo.dot_general %2506, %2516, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2519 = stablehlo.dot_general %2518, %2517, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2520 = stablehlo.convert %arg277 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2521 = stablehlo.dot_general %2506, %2520, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2522 = stablehlo.add %2519, %2521 : tensor<1x8x1024xbf16>
    %2523 = stablehlo.reshape %2513 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2524 = stablehlo.convert %2523 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2525 = stablehlo.multiply %2524, %2524 : tensor<1x8x16x128xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2526 = stablehlo.reduce(%2525 init: %cst_109) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2527 = stablehlo.broadcast_in_dim %2526, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2528 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2529 = stablehlo.divide %2527, %2528 : tensor<1x8x16x1xf32>
    %2530 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2531 = stablehlo.add %2529, %2530 : tensor<1x8x16x1xf32>
    %2532 = stablehlo.rsqrt %2531 : tensor<1x8x16x1xf32>
    %2533 = stablehlo.broadcast_in_dim %2532, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2534 = stablehlo.multiply %2524, %2533 : tensor<1x8x16x128xf32>
    %2535 = stablehlo.convert %2534 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2536 = stablehlo.convert %arg275 : (tensor<128xf32>) -> tensor<128xbf16>
    %2537 = stablehlo.broadcast_in_dim %2536, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2538 = stablehlo.broadcast_in_dim %2537, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2539 = stablehlo.multiply %2538, %2535 : tensor<1x8x16x128xbf16>
    %2540 = stablehlo.reshape %2515 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2541 = stablehlo.convert %2540 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2542 = stablehlo.multiply %2541, %2541 : tensor<1x8x8x128xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2543 = stablehlo.reduce(%2542 init: %cst_110) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2544 = stablehlo.broadcast_in_dim %2543, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2545 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2546 = stablehlo.divide %2544, %2545 : tensor<1x8x8x1xf32>
    %2547 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2548 = stablehlo.add %2546, %2547 : tensor<1x8x8x1xf32>
    %2549 = stablehlo.rsqrt %2548 : tensor<1x8x8x1xf32>
    %2550 = stablehlo.broadcast_in_dim %2549, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2551 = stablehlo.multiply %2541, %2550 : tensor<1x8x8x128xf32>
    %2552 = stablehlo.convert %2551 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2553 = stablehlo.convert %arg272 : (tensor<128xf32>) -> tensor<128xbf16>
    %2554 = stablehlo.broadcast_in_dim %2553, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2555 = stablehlo.broadcast_in_dim %2554, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2556 = stablehlo.multiply %2555, %2552 : tensor<1x8x8x128xbf16>
    %2557 = stablehlo.reshape %2522 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2558 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2559 = stablehlo.compare  LT, %7, %2558,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2560 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2561 = stablehlo.add %7, %2560 : tensor<1x8xi32>
    %2562 = stablehlo.select %2559, %2561, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2563 = stablehlo.broadcast_in_dim %2562, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2564 = "stablehlo.gather"(%26, %2563) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2565 = stablehlo.slice %2564 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2566 = stablehlo.slice %2564 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2567 = stablehlo.broadcast_in_dim %2565, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2568 = stablehlo.convert %2567 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2569 = stablehlo.broadcast_in_dim %2566, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2570 = stablehlo.convert %2569 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2571 = stablehlo.slice %2539 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2572 = stablehlo.slice %2539 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2573 = stablehlo.broadcast_in_dim %2568, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2574 = stablehlo.multiply %2571, %2573 : tensor<1x8x16x64xbf16>
    %2575 = stablehlo.broadcast_in_dim %2570, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2576 = stablehlo.multiply %2572, %2575 : tensor<1x8x16x64xbf16>
    %2577 = stablehlo.subtract %2574, %2576 : tensor<1x8x16x64xbf16>
    %2578 = stablehlo.broadcast_in_dim %2568, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2579 = stablehlo.multiply %2572, %2578 : tensor<1x8x16x64xbf16>
    %2580 = stablehlo.broadcast_in_dim %2570, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2581 = stablehlo.multiply %2571, %2580 : tensor<1x8x16x64xbf16>
    %2582 = stablehlo.add %2579, %2581 : tensor<1x8x16x64xbf16>
    %2583 = stablehlo.concatenate %2577, %2582, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2584 = stablehlo.broadcast_in_dim %2565, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2585 = stablehlo.convert %2584 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2586 = stablehlo.broadcast_in_dim %2566, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2587 = stablehlo.convert %2586 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2588 = stablehlo.slice %2556 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2589 = stablehlo.slice %2556 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2590 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2591 = stablehlo.multiply %2588, %2590 : tensor<1x8x8x64xbf16>
    %2592 = stablehlo.broadcast_in_dim %2587, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2593 = stablehlo.multiply %2589, %2592 : tensor<1x8x8x64xbf16>
    %2594 = stablehlo.subtract %2591, %2593 : tensor<1x8x8x64xbf16>
    %2595 = stablehlo.broadcast_in_dim %2585, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2596 = stablehlo.multiply %2589, %2595 : tensor<1x8x8x64xbf16>
    %2597 = stablehlo.broadcast_in_dim %2587, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2598 = stablehlo.multiply %2588, %2597 : tensor<1x8x8x64xbf16>
    %2599 = stablehlo.add %2596, %2598 : tensor<1x8x8x64xbf16>
    %2600 = stablehlo.concatenate %2594, %2599, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2601 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2602 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2603 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2604 = stablehlo.and %2603, %2602 : tensor<1x1x8x8xi1>
    %2605 = stablehlo.convert %2604 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2606 = sdy.sharding_constraint %2583 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2607 = sdy.sharding_constraint %2600 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2608 = sdy.sharding_constraint %2557 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2609 = sdy.sharding_constraint %2605 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2610 = stablehlo.reshape %2606 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2611 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2612 = stablehlo.multiply %2610, %2611 : tensor<1x8x8x2x128xbf16>
    %2613 = stablehlo.dot_general %2607, %2612, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2614 = stablehlo.transpose %2613, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2615 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2616 = stablehlo.compare  NE, %2609, %2615,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2617 = stablehlo.convert %2616 : tensor<1x1x8x8xi1>
    %2618 = stablehlo.broadcast_in_dim %2617, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2619 = stablehlo.reshape %2618 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2620 = call @_where_91(%2619, %2614, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2621 = stablehlo.convert %2620 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_112 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2622 = stablehlo.reduce(%2621 init: %cst_112) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2623 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2624 = stablehlo.maximum %2623, %2622 : tensor<1x8x2x8xf32>
    %2625 = stablehlo.broadcast_in_dim %2624, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2626 = stablehlo.broadcast_in_dim %2625, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2627 = stablehlo.subtract %2621, %2626 : tensor<1x8x2x8x8xf32>
    %2628 = stablehlo.exponential %2627 : tensor<1x8x2x8x8xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2629 = stablehlo.reduce(%2628 init: %cst_113) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2630 = stablehlo.broadcast_in_dim %2629, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2631 = stablehlo.broadcast_in_dim %2630, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2632 = stablehlo.divide %2628, %2631 : tensor<1x8x2x8x8xf32>
    %2633 = stablehlo.convert %2632 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2634 = stablehlo.dot_general %2608, %2633, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2635 = stablehlo.transpose %2634, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2636 = stablehlo.reshape %2635 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2637 = sdy.sharding_constraint %2636 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2638 = stablehlo.reshape %2637 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2639 = stablehlo.convert %arg274 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2640 = stablehlo.dot_general %2638, %2639, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2641 = stablehlo.add %2490, %2640 : tensor<1x8x1024xbf16>
    %2642 = stablehlo.convert %2641 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2643 = stablehlo.multiply %2642, %2642 : tensor<1x8x1024xf32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2644 = stablehlo.reduce(%2643 init: %cst_114) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2645 = stablehlo.broadcast_in_dim %2644, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2646 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2647 = stablehlo.divide %2645, %2646 : tensor<1x8x1xf32>
    %2648 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2649 = stablehlo.add %2647, %2648 : tensor<1x8x1xf32>
    %2650 = stablehlo.rsqrt %2649 : tensor<1x8x1xf32>
    %2651 = stablehlo.broadcast_in_dim %2650, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2652 = stablehlo.multiply %2642, %2651 : tensor<1x8x1024xf32>
    %2653 = stablehlo.convert %2652 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2654 = stablehlo.convert %arg271 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2655 = stablehlo.broadcast_in_dim %2654, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2656 = stablehlo.broadcast_in_dim %2655, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2657 = stablehlo.multiply %2656, %2653 : tensor<1x8x1024xbf16>
    %2658 = stablehlo.convert %arg269 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2659 = stablehlo.dot_general %2657, %2658, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2660 = call @silu(%2659) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2661 = stablehlo.convert %arg270 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2662 = stablehlo.dot_general %2657, %2661, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2663 = stablehlo.multiply %2660, %2662 : tensor<1x8x3072xbf16>
    %2664 = stablehlo.convert %arg268 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2665 = stablehlo.dot_general %2663, %2664, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2666 = stablehlo.add %2641, %2665 : tensor<1x8x1024xbf16>
    %2667 = stablehlo.convert %2666 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2668 = stablehlo.multiply %2667, %2667 : tensor<1x8x1024xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2669 = stablehlo.reduce(%2668 init: %cst_115) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2670 = stablehlo.broadcast_in_dim %2669, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2671 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2672 = stablehlo.divide %2670, %2671 : tensor<1x8x1xf32>
    %2673 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2674 = stablehlo.add %2672, %2673 : tensor<1x8x1xf32>
    %2675 = stablehlo.rsqrt %2674 : tensor<1x8x1xf32>
    %2676 = stablehlo.broadcast_in_dim %2675, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2677 = stablehlo.multiply %2667, %2676 : tensor<1x8x1024xf32>
    %2678 = stablehlo.convert %2677 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2679 = stablehlo.convert %arg278 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2680 = stablehlo.broadcast_in_dim %2679, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2681 = stablehlo.broadcast_in_dim %2680, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2682 = stablehlo.multiply %2681, %2678 : tensor<1x8x1024xbf16>
    %2683 = stablehlo.convert %arg60 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2684 = stablehlo.convert %arg61 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %2685 = stablehlo.dot_general %2682, %2683, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2686 = stablehlo.dot_general %2685, %2684, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2687 = stablehlo.convert %arg287 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2688 = stablehlo.dot_general %2682, %2687, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2689 = stablehlo.add %2686, %2688 : tensor<1x8x2048xbf16>
    %2690 = stablehlo.convert %arg284 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2691 = stablehlo.dot_general %2682, %2690, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2692 = stablehlo.convert %arg62 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2693 = stablehlo.convert %arg63 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %2694 = stablehlo.dot_general %2682, %2692, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2695 = stablehlo.dot_general %2694, %2693, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2696 = stablehlo.convert %arg288 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2697 = stablehlo.dot_general %2682, %2696, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2698 = stablehlo.add %2695, %2697 : tensor<1x8x1024xbf16>
    %2699 = stablehlo.reshape %2689 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2700 = stablehlo.convert %2699 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2701 = stablehlo.multiply %2700, %2700 : tensor<1x8x16x128xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2702 = stablehlo.reduce(%2701 init: %cst_116) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2703 = stablehlo.broadcast_in_dim %2702, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2704 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2705 = stablehlo.divide %2703, %2704 : tensor<1x8x16x1xf32>
    %2706 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2707 = stablehlo.add %2705, %2706 : tensor<1x8x16x1xf32>
    %2708 = stablehlo.rsqrt %2707 : tensor<1x8x16x1xf32>
    %2709 = stablehlo.broadcast_in_dim %2708, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2710 = stablehlo.multiply %2700, %2709 : tensor<1x8x16x128xf32>
    %2711 = stablehlo.convert %2710 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2712 = stablehlo.convert %arg286 : (tensor<128xf32>) -> tensor<128xbf16>
    %2713 = stablehlo.broadcast_in_dim %2712, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2714 = stablehlo.broadcast_in_dim %2713, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2715 = stablehlo.multiply %2714, %2711 : tensor<1x8x16x128xbf16>
    %2716 = stablehlo.reshape %2691 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2717 = stablehlo.convert %2716 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2718 = stablehlo.multiply %2717, %2717 : tensor<1x8x8x128xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2719 = stablehlo.reduce(%2718 init: %cst_117) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2720 = stablehlo.broadcast_in_dim %2719, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2721 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2722 = stablehlo.divide %2720, %2721 : tensor<1x8x8x1xf32>
    %2723 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2724 = stablehlo.add %2722, %2723 : tensor<1x8x8x1xf32>
    %2725 = stablehlo.rsqrt %2724 : tensor<1x8x8x1xf32>
    %2726 = stablehlo.broadcast_in_dim %2725, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2727 = stablehlo.multiply %2717, %2726 : tensor<1x8x8x128xf32>
    %2728 = stablehlo.convert %2727 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2729 = stablehlo.convert %arg283 : (tensor<128xf32>) -> tensor<128xbf16>
    %2730 = stablehlo.broadcast_in_dim %2729, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2731 = stablehlo.broadcast_in_dim %2730, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2732 = stablehlo.multiply %2731, %2728 : tensor<1x8x8x128xbf16>
    %2733 = stablehlo.reshape %2698 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2734 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2735 = stablehlo.compare  LT, %7, %2734,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2736 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2737 = stablehlo.add %7, %2736 : tensor<1x8xi32>
    %2738 = stablehlo.select %2735, %2737, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2739 = stablehlo.broadcast_in_dim %2738, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2740 = "stablehlo.gather"(%26, %2739) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2741 = stablehlo.slice %2740 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2742 = stablehlo.slice %2740 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2743 = stablehlo.broadcast_in_dim %2741, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2744 = stablehlo.convert %2743 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2745 = stablehlo.broadcast_in_dim %2742, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2746 = stablehlo.convert %2745 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2747 = stablehlo.slice %2715 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2748 = stablehlo.slice %2715 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2749 = stablehlo.broadcast_in_dim %2744, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2750 = stablehlo.multiply %2747, %2749 : tensor<1x8x16x64xbf16>
    %2751 = stablehlo.broadcast_in_dim %2746, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2752 = stablehlo.multiply %2748, %2751 : tensor<1x8x16x64xbf16>
    %2753 = stablehlo.subtract %2750, %2752 : tensor<1x8x16x64xbf16>
    %2754 = stablehlo.broadcast_in_dim %2744, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2755 = stablehlo.multiply %2748, %2754 : tensor<1x8x16x64xbf16>
    %2756 = stablehlo.broadcast_in_dim %2746, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2757 = stablehlo.multiply %2747, %2756 : tensor<1x8x16x64xbf16>
    %2758 = stablehlo.add %2755, %2757 : tensor<1x8x16x64xbf16>
    %2759 = stablehlo.concatenate %2753, %2758, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2760 = stablehlo.broadcast_in_dim %2741, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2761 = stablehlo.convert %2760 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2762 = stablehlo.broadcast_in_dim %2742, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2763 = stablehlo.convert %2762 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2764 = stablehlo.slice %2732 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2765 = stablehlo.slice %2732 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2766 = stablehlo.broadcast_in_dim %2761, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2767 = stablehlo.multiply %2764, %2766 : tensor<1x8x8x64xbf16>
    %2768 = stablehlo.broadcast_in_dim %2763, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2769 = stablehlo.multiply %2765, %2768 : tensor<1x8x8x64xbf16>
    %2770 = stablehlo.subtract %2767, %2769 : tensor<1x8x8x64xbf16>
    %2771 = stablehlo.broadcast_in_dim %2761, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2772 = stablehlo.multiply %2765, %2771 : tensor<1x8x8x64xbf16>
    %2773 = stablehlo.broadcast_in_dim %2763, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2774 = stablehlo.multiply %2764, %2773 : tensor<1x8x8x64xbf16>
    %2775 = stablehlo.add %2772, %2774 : tensor<1x8x8x64xbf16>
    %2776 = stablehlo.concatenate %2770, %2775, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2777 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2778 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2779 = stablehlo.broadcast_in_dim %2777, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2780 = stablehlo.and %2779, %2778 : tensor<1x1x8x8xi1>
    %2781 = stablehlo.convert %2780 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2782 = sdy.sharding_constraint %2759 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2783 = sdy.sharding_constraint %2776 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2784 = sdy.sharding_constraint %2733 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2785 = sdy.sharding_constraint %2781 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2786 = stablehlo.reshape %2782 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2787 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2788 = stablehlo.multiply %2786, %2787 : tensor<1x8x8x2x128xbf16>
    %2789 = stablehlo.dot_general %2783, %2788, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2790 = stablehlo.transpose %2789, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2791 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2792 = stablehlo.compare  NE, %2785, %2791,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2793 = stablehlo.convert %2792 : tensor<1x1x8x8xi1>
    %2794 = stablehlo.broadcast_in_dim %2793, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2795 = stablehlo.reshape %2794 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2796 = call @_where_91(%2795, %2790, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2797 = stablehlo.convert %2796 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_119 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2798 = stablehlo.reduce(%2797 init: %cst_119) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2799 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2800 = stablehlo.maximum %2799, %2798 : tensor<1x8x2x8xf32>
    %2801 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2802 = stablehlo.broadcast_in_dim %2801, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2803 = stablehlo.subtract %2797, %2802 : tensor<1x8x2x8x8xf32>
    %2804 = stablehlo.exponential %2803 : tensor<1x8x2x8x8xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2805 = stablehlo.reduce(%2804 init: %cst_120) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2806 = stablehlo.broadcast_in_dim %2805, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2807 = stablehlo.broadcast_in_dim %2806, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2808 = stablehlo.divide %2804, %2807 : tensor<1x8x2x8x8xf32>
    %2809 = stablehlo.convert %2808 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2810 = stablehlo.dot_general %2784, %2809, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2811 = stablehlo.transpose %2810, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2812 = stablehlo.reshape %2811 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2813 = sdy.sharding_constraint %2812 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2814 = stablehlo.reshape %2813 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2815 = stablehlo.convert %arg285 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2816 = stablehlo.dot_general %2814, %2815, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2817 = stablehlo.add %2666, %2816 : tensor<1x8x1024xbf16>
    %2818 = stablehlo.convert %2817 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2819 = stablehlo.multiply %2818, %2818 : tensor<1x8x1024xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2820 = stablehlo.reduce(%2819 init: %cst_121) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2821 = stablehlo.broadcast_in_dim %2820, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2822 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2823 = stablehlo.divide %2821, %2822 : tensor<1x8x1xf32>
    %2824 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2825 = stablehlo.add %2823, %2824 : tensor<1x8x1xf32>
    %2826 = stablehlo.rsqrt %2825 : tensor<1x8x1xf32>
    %2827 = stablehlo.broadcast_in_dim %2826, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2828 = stablehlo.multiply %2818, %2827 : tensor<1x8x1024xf32>
    %2829 = stablehlo.convert %2828 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2830 = stablehlo.convert %arg282 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2831 = stablehlo.broadcast_in_dim %2830, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2832 = stablehlo.broadcast_in_dim %2831, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2833 = stablehlo.multiply %2832, %2829 : tensor<1x8x1024xbf16>
    %2834 = stablehlo.convert %arg280 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2835 = stablehlo.dot_general %2833, %2834, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2836 = call @silu(%2835) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2837 = stablehlo.convert %arg281 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2838 = stablehlo.dot_general %2833, %2837, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2839 = stablehlo.multiply %2836, %2838 : tensor<1x8x3072xbf16>
    %2840 = stablehlo.convert %arg279 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2841 = stablehlo.dot_general %2839, %2840, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2842 = stablehlo.add %2817, %2841 : tensor<1x8x1024xbf16>
    %2843 = stablehlo.convert %2842 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2844 = stablehlo.multiply %2843, %2843 : tensor<1x8x1024xf32>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2845 = stablehlo.reduce(%2844 init: %cst_122) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2846 = stablehlo.broadcast_in_dim %2845, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2847 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2848 = stablehlo.divide %2846, %2847 : tensor<1x8x1xf32>
    %2849 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2850 = stablehlo.add %2848, %2849 : tensor<1x8x1xf32>
    %2851 = stablehlo.rsqrt %2850 : tensor<1x8x1xf32>
    %2852 = stablehlo.broadcast_in_dim %2851, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2853 = stablehlo.multiply %2843, %2852 : tensor<1x8x1024xf32>
    %2854 = stablehlo.convert %2853 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2855 = stablehlo.convert %arg289 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2856 = stablehlo.broadcast_in_dim %2855, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2857 = stablehlo.broadcast_in_dim %2856, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2858 = stablehlo.multiply %2857, %2854 : tensor<1x8x1024xbf16>
    %2859 = stablehlo.convert %arg64 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2860 = stablehlo.convert %arg65 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %2861 = stablehlo.dot_general %2858, %2859, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2862 = stablehlo.dot_general %2861, %2860, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2863 = stablehlo.convert %arg298 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2864 = stablehlo.dot_general %2858, %2863, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2865 = stablehlo.add %2862, %2864 : tensor<1x8x2048xbf16>
    %2866 = stablehlo.convert %arg295 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2867 = stablehlo.dot_general %2858, %2866, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2868 = stablehlo.convert %arg66 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %2869 = stablehlo.convert %arg67 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %2870 = stablehlo.dot_general %2858, %2868, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %2871 = stablehlo.dot_general %2870, %2869, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2872 = stablehlo.convert %arg299 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2873 = stablehlo.dot_general %2858, %2872, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2874 = stablehlo.add %2871, %2873 : tensor<1x8x1024xbf16>
    %2875 = stablehlo.reshape %2865 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2876 = stablehlo.convert %2875 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2877 = stablehlo.multiply %2876, %2876 : tensor<1x8x16x128xf32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2878 = stablehlo.reduce(%2877 init: %cst_123) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2879 = stablehlo.broadcast_in_dim %2878, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2880 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2881 = stablehlo.divide %2879, %2880 : tensor<1x8x16x1xf32>
    %2882 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2883 = stablehlo.add %2881, %2882 : tensor<1x8x16x1xf32>
    %2884 = stablehlo.rsqrt %2883 : tensor<1x8x16x1xf32>
    %2885 = stablehlo.broadcast_in_dim %2884, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2886 = stablehlo.multiply %2876, %2885 : tensor<1x8x16x128xf32>
    %2887 = stablehlo.convert %2886 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2888 = stablehlo.convert %arg297 : (tensor<128xf32>) -> tensor<128xbf16>
    %2889 = stablehlo.broadcast_in_dim %2888, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2890 = stablehlo.broadcast_in_dim %2889, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2891 = stablehlo.multiply %2890, %2887 : tensor<1x8x16x128xbf16>
    %2892 = stablehlo.reshape %2867 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2893 = stablehlo.convert %2892 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2894 = stablehlo.multiply %2893, %2893 : tensor<1x8x8x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2895 = stablehlo.reduce(%2894 init: %cst_124) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2896 = stablehlo.broadcast_in_dim %2895, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2897 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2898 = stablehlo.divide %2896, %2897 : tensor<1x8x8x1xf32>
    %2899 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2900 = stablehlo.add %2898, %2899 : tensor<1x8x8x1xf32>
    %2901 = stablehlo.rsqrt %2900 : tensor<1x8x8x1xf32>
    %2902 = stablehlo.broadcast_in_dim %2901, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2903 = stablehlo.multiply %2893, %2902 : tensor<1x8x8x128xf32>
    %2904 = stablehlo.convert %2903 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2905 = stablehlo.convert %arg294 : (tensor<128xf32>) -> tensor<128xbf16>
    %2906 = stablehlo.broadcast_in_dim %2905, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2907 = stablehlo.broadcast_in_dim %2906, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2908 = stablehlo.multiply %2907, %2904 : tensor<1x8x8x128xbf16>
    %2909 = stablehlo.reshape %2874 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2910 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2911 = stablehlo.compare  LT, %7, %2910,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2912 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2913 = stablehlo.add %7, %2912 : tensor<1x8xi32>
    %2914 = stablehlo.select %2911, %2913, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2915 = stablehlo.broadcast_in_dim %2914, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2916 = "stablehlo.gather"(%26, %2915) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2917 = stablehlo.slice %2916 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2918 = stablehlo.slice %2916 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2919 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2920 = stablehlo.convert %2919 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2921 = stablehlo.broadcast_in_dim %2918, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2922 = stablehlo.convert %2921 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2923 = stablehlo.slice %2891 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2924 = stablehlo.slice %2891 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2925 = stablehlo.broadcast_in_dim %2920, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2926 = stablehlo.multiply %2923, %2925 : tensor<1x8x16x64xbf16>
    %2927 = stablehlo.broadcast_in_dim %2922, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2928 = stablehlo.multiply %2924, %2927 : tensor<1x8x16x64xbf16>
    %2929 = stablehlo.subtract %2926, %2928 : tensor<1x8x16x64xbf16>
    %2930 = stablehlo.broadcast_in_dim %2920, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2931 = stablehlo.multiply %2924, %2930 : tensor<1x8x16x64xbf16>
    %2932 = stablehlo.broadcast_in_dim %2922, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2933 = stablehlo.multiply %2923, %2932 : tensor<1x8x16x64xbf16>
    %2934 = stablehlo.add %2931, %2933 : tensor<1x8x16x64xbf16>
    %2935 = stablehlo.concatenate %2929, %2934, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2936 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2937 = stablehlo.convert %2936 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2938 = stablehlo.broadcast_in_dim %2918, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2939 = stablehlo.convert %2938 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2940 = stablehlo.slice %2908 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2941 = stablehlo.slice %2908 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2942 = stablehlo.broadcast_in_dim %2937, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2943 = stablehlo.multiply %2940, %2942 : tensor<1x8x8x64xbf16>
    %2944 = stablehlo.broadcast_in_dim %2939, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2945 = stablehlo.multiply %2941, %2944 : tensor<1x8x8x64xbf16>
    %2946 = stablehlo.subtract %2943, %2945 : tensor<1x8x8x64xbf16>
    %2947 = stablehlo.broadcast_in_dim %2937, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2948 = stablehlo.multiply %2941, %2947 : tensor<1x8x8x64xbf16>
    %2949 = stablehlo.broadcast_in_dim %2939, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2950 = stablehlo.multiply %2940, %2949 : tensor<1x8x8x64xbf16>
    %2951 = stablehlo.add %2948, %2950 : tensor<1x8x8x64xbf16>
    %2952 = stablehlo.concatenate %2946, %2951, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2953 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2954 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2955 = stablehlo.broadcast_in_dim %2953, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2956 = stablehlo.and %2955, %2954 : tensor<1x1x8x8xi1>
    %2957 = stablehlo.convert %2956 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2958 = sdy.sharding_constraint %2935 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2959 = sdy.sharding_constraint %2952 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2960 = sdy.sharding_constraint %2909 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2961 = sdy.sharding_constraint %2957 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2962 = stablehlo.reshape %2958 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2963 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2964 = stablehlo.multiply %2962, %2963 : tensor<1x8x8x2x128xbf16>
    %2965 = stablehlo.dot_general %2959, %2964, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2966 = stablehlo.transpose %2965, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2967 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2968 = stablehlo.compare  NE, %2961, %2967,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2969 = stablehlo.convert %2968 : tensor<1x1x8x8xi1>
    %2970 = stablehlo.broadcast_in_dim %2969, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2971 = stablehlo.reshape %2970 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2972 = call @_where_91(%2971, %2966, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2973 = stablehlo.convert %2972 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_126 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2974 = stablehlo.reduce(%2973 init: %cst_126) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2975 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2976 = stablehlo.maximum %2975, %2974 : tensor<1x8x2x8xf32>
    %2977 = stablehlo.broadcast_in_dim %2976, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2978 = stablehlo.broadcast_in_dim %2977, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2979 = stablehlo.subtract %2973, %2978 : tensor<1x8x2x8x8xf32>
    %2980 = stablehlo.exponential %2979 : tensor<1x8x2x8x8xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2981 = stablehlo.reduce(%2980 init: %cst_127) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2982 = stablehlo.broadcast_in_dim %2981, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2983 = stablehlo.broadcast_in_dim %2982, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2984 = stablehlo.divide %2980, %2983 : tensor<1x8x2x8x8xf32>
    %2985 = stablehlo.convert %2984 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2986 = stablehlo.dot_general %2960, %2985, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2987 = stablehlo.transpose %2986, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2988 = stablehlo.reshape %2987 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2989 = sdy.sharding_constraint %2988 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2990 = stablehlo.reshape %2989 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2991 = stablehlo.convert %arg296 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2992 = stablehlo.dot_general %2990, %2991, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2993 = stablehlo.add %2842, %2992 : tensor<1x8x1024xbf16>
    %2994 = stablehlo.convert %2993 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2995 = stablehlo.multiply %2994, %2994 : tensor<1x8x1024xf32>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2996 = stablehlo.reduce(%2995 init: %cst_128) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2997 = stablehlo.broadcast_in_dim %2996, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2998 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2999 = stablehlo.divide %2997, %2998 : tensor<1x8x1xf32>
    %3000 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3001 = stablehlo.add %2999, %3000 : tensor<1x8x1xf32>
    %3002 = stablehlo.rsqrt %3001 : tensor<1x8x1xf32>
    %3003 = stablehlo.broadcast_in_dim %3002, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3004 = stablehlo.multiply %2994, %3003 : tensor<1x8x1024xf32>
    %3005 = stablehlo.convert %3004 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3006 = stablehlo.convert %arg293 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3007 = stablehlo.broadcast_in_dim %3006, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3008 = stablehlo.broadcast_in_dim %3007, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3009 = stablehlo.multiply %3008, %3005 : tensor<1x8x1024xbf16>
    %3010 = stablehlo.convert %arg291 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3011 = stablehlo.dot_general %3009, %3010, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3012 = call @silu(%3011) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3013 = stablehlo.convert %arg292 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3014 = stablehlo.dot_general %3009, %3013, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3015 = stablehlo.multiply %3012, %3014 : tensor<1x8x3072xbf16>
    %3016 = stablehlo.convert %arg290 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3017 = stablehlo.dot_general %3015, %3016, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3018 = stablehlo.add %2993, %3017 : tensor<1x8x1024xbf16>
    %3019 = stablehlo.convert %3018 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3020 = stablehlo.multiply %3019, %3019 : tensor<1x8x1024xf32>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3021 = stablehlo.reduce(%3020 init: %cst_129) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3022 = stablehlo.broadcast_in_dim %3021, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3023 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3024 = stablehlo.divide %3022, %3023 : tensor<1x8x1xf32>
    %3025 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3026 = stablehlo.add %3024, %3025 : tensor<1x8x1xf32>
    %3027 = stablehlo.rsqrt %3026 : tensor<1x8x1xf32>
    %3028 = stablehlo.broadcast_in_dim %3027, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3029 = stablehlo.multiply %3019, %3028 : tensor<1x8x1024xf32>
    %3030 = stablehlo.convert %3029 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3031 = stablehlo.convert %arg300 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3032 = stablehlo.broadcast_in_dim %3031, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3033 = stablehlo.broadcast_in_dim %3032, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3034 = stablehlo.multiply %3033, %3030 : tensor<1x8x1024xbf16>
    %3035 = stablehlo.convert %arg68 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3036 = stablehlo.convert %arg69 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3037 = stablehlo.dot_general %3034, %3035, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3038 = stablehlo.dot_general %3037, %3036, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3039 = stablehlo.convert %arg309 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3040 = stablehlo.dot_general %3034, %3039, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3041 = stablehlo.add %3038, %3040 : tensor<1x8x2048xbf16>
    %3042 = stablehlo.convert %arg306 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3043 = stablehlo.dot_general %3034, %3042, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3044 = stablehlo.convert %arg70 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3045 = stablehlo.convert %arg71 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3046 = stablehlo.dot_general %3034, %3044, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3047 = stablehlo.dot_general %3046, %3045, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3048 = stablehlo.convert %arg310 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3049 = stablehlo.dot_general %3034, %3048, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3050 = stablehlo.add %3047, %3049 : tensor<1x8x1024xbf16>
    %3051 = stablehlo.reshape %3041 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3052 = stablehlo.convert %3051 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3053 = stablehlo.multiply %3052, %3052 : tensor<1x8x16x128xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3054 = stablehlo.reduce(%3053 init: %cst_130) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3055 = stablehlo.broadcast_in_dim %3054, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3056 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3057 = stablehlo.divide %3055, %3056 : tensor<1x8x16x1xf32>
    %3058 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3059 = stablehlo.add %3057, %3058 : tensor<1x8x16x1xf32>
    %3060 = stablehlo.rsqrt %3059 : tensor<1x8x16x1xf32>
    %3061 = stablehlo.broadcast_in_dim %3060, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3062 = stablehlo.multiply %3052, %3061 : tensor<1x8x16x128xf32>
    %3063 = stablehlo.convert %3062 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3064 = stablehlo.convert %arg308 : (tensor<128xf32>) -> tensor<128xbf16>
    %3065 = stablehlo.broadcast_in_dim %3064, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3066 = stablehlo.broadcast_in_dim %3065, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3067 = stablehlo.multiply %3066, %3063 : tensor<1x8x16x128xbf16>
    %3068 = stablehlo.reshape %3043 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3069 = stablehlo.convert %3068 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3070 = stablehlo.multiply %3069, %3069 : tensor<1x8x8x128xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3071 = stablehlo.reduce(%3070 init: %cst_131) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3072 = stablehlo.broadcast_in_dim %3071, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3073 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3074 = stablehlo.divide %3072, %3073 : tensor<1x8x8x1xf32>
    %3075 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3076 = stablehlo.add %3074, %3075 : tensor<1x8x8x1xf32>
    %3077 = stablehlo.rsqrt %3076 : tensor<1x8x8x1xf32>
    %3078 = stablehlo.broadcast_in_dim %3077, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3079 = stablehlo.multiply %3069, %3078 : tensor<1x8x8x128xf32>
    %3080 = stablehlo.convert %3079 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3081 = stablehlo.convert %arg305 : (tensor<128xf32>) -> tensor<128xbf16>
    %3082 = stablehlo.broadcast_in_dim %3081, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3083 = stablehlo.broadcast_in_dim %3082, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3084 = stablehlo.multiply %3083, %3080 : tensor<1x8x8x128xbf16>
    %3085 = stablehlo.reshape %3050 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3086 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3087 = stablehlo.compare  LT, %7, %3086,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3088 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3089 = stablehlo.add %7, %3088 : tensor<1x8xi32>
    %3090 = stablehlo.select %3087, %3089, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3091 = stablehlo.broadcast_in_dim %3090, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3092 = "stablehlo.gather"(%26, %3091) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3093 = stablehlo.slice %3092 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3094 = stablehlo.slice %3092 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3095 = stablehlo.broadcast_in_dim %3093, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3096 = stablehlo.convert %3095 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3097 = stablehlo.broadcast_in_dim %3094, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3098 = stablehlo.convert %3097 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3099 = stablehlo.slice %3067 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3100 = stablehlo.slice %3067 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3101 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3102 = stablehlo.multiply %3099, %3101 : tensor<1x8x16x64xbf16>
    %3103 = stablehlo.broadcast_in_dim %3098, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3104 = stablehlo.multiply %3100, %3103 : tensor<1x8x16x64xbf16>
    %3105 = stablehlo.subtract %3102, %3104 : tensor<1x8x16x64xbf16>
    %3106 = stablehlo.broadcast_in_dim %3096, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3107 = stablehlo.multiply %3100, %3106 : tensor<1x8x16x64xbf16>
    %3108 = stablehlo.broadcast_in_dim %3098, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3109 = stablehlo.multiply %3099, %3108 : tensor<1x8x16x64xbf16>
    %3110 = stablehlo.add %3107, %3109 : tensor<1x8x16x64xbf16>
    %3111 = stablehlo.concatenate %3105, %3110, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3112 = stablehlo.broadcast_in_dim %3093, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3113 = stablehlo.convert %3112 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3114 = stablehlo.broadcast_in_dim %3094, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3115 = stablehlo.convert %3114 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3116 = stablehlo.slice %3084 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3117 = stablehlo.slice %3084 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3118 = stablehlo.broadcast_in_dim %3113, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3119 = stablehlo.multiply %3116, %3118 : tensor<1x8x8x64xbf16>
    %3120 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3121 = stablehlo.multiply %3117, %3120 : tensor<1x8x8x64xbf16>
    %3122 = stablehlo.subtract %3119, %3121 : tensor<1x8x8x64xbf16>
    %3123 = stablehlo.broadcast_in_dim %3113, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3124 = stablehlo.multiply %3117, %3123 : tensor<1x8x8x64xbf16>
    %3125 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3126 = stablehlo.multiply %3116, %3125 : tensor<1x8x8x64xbf16>
    %3127 = stablehlo.add %3124, %3126 : tensor<1x8x8x64xbf16>
    %3128 = stablehlo.concatenate %3122, %3127, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3129 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3130 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3131 = stablehlo.broadcast_in_dim %3129, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3132 = stablehlo.and %3131, %3130 : tensor<1x1x8x8xi1>
    %3133 = stablehlo.convert %3132 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3134 = sdy.sharding_constraint %3111 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3135 = sdy.sharding_constraint %3128 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3136 = sdy.sharding_constraint %3085 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3137 = sdy.sharding_constraint %3133 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3138 = stablehlo.reshape %3134 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3139 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3140 = stablehlo.multiply %3138, %3139 : tensor<1x8x8x2x128xbf16>
    %3141 = stablehlo.dot_general %3135, %3140, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3142 = stablehlo.transpose %3141, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3143 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3144 = stablehlo.compare  NE, %3137, %3143,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3145 = stablehlo.convert %3144 : tensor<1x1x8x8xi1>
    %3146 = stablehlo.broadcast_in_dim %3145, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3147 = stablehlo.reshape %3146 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3148 = call @_where_91(%3147, %3142, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3149 = stablehlo.convert %3148 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_133 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3150 = stablehlo.reduce(%3149 init: %cst_133) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3151 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3152 = stablehlo.maximum %3151, %3150 : tensor<1x8x2x8xf32>
    %3153 = stablehlo.broadcast_in_dim %3152, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3154 = stablehlo.broadcast_in_dim %3153, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3155 = stablehlo.subtract %3149, %3154 : tensor<1x8x2x8x8xf32>
    %3156 = stablehlo.exponential %3155 : tensor<1x8x2x8x8xf32>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3157 = stablehlo.reduce(%3156 init: %cst_134) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3158 = stablehlo.broadcast_in_dim %3157, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3159 = stablehlo.broadcast_in_dim %3158, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3160 = stablehlo.divide %3156, %3159 : tensor<1x8x2x8x8xf32>
    %3161 = stablehlo.convert %3160 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3162 = stablehlo.dot_general %3136, %3161, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3163 = stablehlo.transpose %3162, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3164 = stablehlo.reshape %3163 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3165 = sdy.sharding_constraint %3164 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3166 = stablehlo.reshape %3165 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3167 = stablehlo.convert %arg307 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3168 = stablehlo.dot_general %3166, %3167, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3169 = stablehlo.add %3018, %3168 : tensor<1x8x1024xbf16>
    %3170 = stablehlo.convert %3169 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3171 = stablehlo.multiply %3170, %3170 : tensor<1x8x1024xf32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3172 = stablehlo.reduce(%3171 init: %cst_135) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3173 = stablehlo.broadcast_in_dim %3172, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3174 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3175 = stablehlo.divide %3173, %3174 : tensor<1x8x1xf32>
    %3176 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3177 = stablehlo.add %3175, %3176 : tensor<1x8x1xf32>
    %3178 = stablehlo.rsqrt %3177 : tensor<1x8x1xf32>
    %3179 = stablehlo.broadcast_in_dim %3178, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3180 = stablehlo.multiply %3170, %3179 : tensor<1x8x1024xf32>
    %3181 = stablehlo.convert %3180 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3182 = stablehlo.convert %arg304 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3183 = stablehlo.broadcast_in_dim %3182, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3184 = stablehlo.broadcast_in_dim %3183, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3185 = stablehlo.multiply %3184, %3181 : tensor<1x8x1024xbf16>
    %3186 = stablehlo.convert %arg302 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3187 = stablehlo.dot_general %3185, %3186, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3188 = call @silu(%3187) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3189 = stablehlo.convert %arg303 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3190 = stablehlo.dot_general %3185, %3189, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3191 = stablehlo.multiply %3188, %3190 : tensor<1x8x3072xbf16>
    %3192 = stablehlo.convert %arg301 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3193 = stablehlo.dot_general %3191, %3192, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3194 = stablehlo.add %3169, %3193 : tensor<1x8x1024xbf16>
    %3195 = stablehlo.convert %3194 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3196 = stablehlo.multiply %3195, %3195 : tensor<1x8x1024xf32>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3197 = stablehlo.reduce(%3196 init: %cst_136) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3198 = stablehlo.broadcast_in_dim %3197, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3199 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3200 = stablehlo.divide %3198, %3199 : tensor<1x8x1xf32>
    %3201 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3202 = stablehlo.add %3200, %3201 : tensor<1x8x1xf32>
    %3203 = stablehlo.rsqrt %3202 : tensor<1x8x1xf32>
    %3204 = stablehlo.broadcast_in_dim %3203, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3205 = stablehlo.multiply %3195, %3204 : tensor<1x8x1024xf32>
    %3206 = stablehlo.convert %3205 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3207 = stablehlo.convert %arg311 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3208 = stablehlo.broadcast_in_dim %3207, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3209 = stablehlo.broadcast_in_dim %3208, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3210 = stablehlo.multiply %3209, %3206 : tensor<1x8x1024xbf16>
    %3211 = stablehlo.convert %arg72 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3212 = stablehlo.convert %arg73 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3213 = stablehlo.dot_general %3210, %3211, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3214 = stablehlo.dot_general %3213, %3212, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3215 = stablehlo.convert %arg320 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3216 = stablehlo.dot_general %3210, %3215, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3217 = stablehlo.add %3214, %3216 : tensor<1x8x2048xbf16>
    %3218 = stablehlo.convert %arg317 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3219 = stablehlo.dot_general %3210, %3218, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3220 = stablehlo.convert %arg74 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3221 = stablehlo.convert %arg75 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3222 = stablehlo.dot_general %3210, %3220, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3223 = stablehlo.dot_general %3222, %3221, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3224 = stablehlo.convert %arg321 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3225 = stablehlo.dot_general %3210, %3224, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3226 = stablehlo.add %3223, %3225 : tensor<1x8x1024xbf16>
    %3227 = stablehlo.reshape %3217 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3228 = stablehlo.convert %3227 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3229 = stablehlo.multiply %3228, %3228 : tensor<1x8x16x128xf32>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3230 = stablehlo.reduce(%3229 init: %cst_137) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3231 = stablehlo.broadcast_in_dim %3230, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3232 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3233 = stablehlo.divide %3231, %3232 : tensor<1x8x16x1xf32>
    %3234 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3235 = stablehlo.add %3233, %3234 : tensor<1x8x16x1xf32>
    %3236 = stablehlo.rsqrt %3235 : tensor<1x8x16x1xf32>
    %3237 = stablehlo.broadcast_in_dim %3236, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3238 = stablehlo.multiply %3228, %3237 : tensor<1x8x16x128xf32>
    %3239 = stablehlo.convert %3238 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3240 = stablehlo.convert %arg319 : (tensor<128xf32>) -> tensor<128xbf16>
    %3241 = stablehlo.broadcast_in_dim %3240, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3242 = stablehlo.broadcast_in_dim %3241, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3243 = stablehlo.multiply %3242, %3239 : tensor<1x8x16x128xbf16>
    %3244 = stablehlo.reshape %3219 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3245 = stablehlo.convert %3244 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3246 = stablehlo.multiply %3245, %3245 : tensor<1x8x8x128xf32>
    %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3247 = stablehlo.reduce(%3246 init: %cst_138) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3248 = stablehlo.broadcast_in_dim %3247, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3249 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3250 = stablehlo.divide %3248, %3249 : tensor<1x8x8x1xf32>
    %3251 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3252 = stablehlo.add %3250, %3251 : tensor<1x8x8x1xf32>
    %3253 = stablehlo.rsqrt %3252 : tensor<1x8x8x1xf32>
    %3254 = stablehlo.broadcast_in_dim %3253, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3255 = stablehlo.multiply %3245, %3254 : tensor<1x8x8x128xf32>
    %3256 = stablehlo.convert %3255 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3257 = stablehlo.convert %arg316 : (tensor<128xf32>) -> tensor<128xbf16>
    %3258 = stablehlo.broadcast_in_dim %3257, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3259 = stablehlo.broadcast_in_dim %3258, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3260 = stablehlo.multiply %3259, %3256 : tensor<1x8x8x128xbf16>
    %3261 = stablehlo.reshape %3226 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3262 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3263 = stablehlo.compare  LT, %7, %3262,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3264 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3265 = stablehlo.add %7, %3264 : tensor<1x8xi32>
    %3266 = stablehlo.select %3263, %3265, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3267 = stablehlo.broadcast_in_dim %3266, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3268 = "stablehlo.gather"(%26, %3267) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3269 = stablehlo.slice %3268 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3270 = stablehlo.slice %3268 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3271 = stablehlo.broadcast_in_dim %3269, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3272 = stablehlo.convert %3271 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3273 = stablehlo.broadcast_in_dim %3270, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3274 = stablehlo.convert %3273 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3275 = stablehlo.slice %3243 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3276 = stablehlo.slice %3243 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3277 = stablehlo.broadcast_in_dim %3272, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3278 = stablehlo.multiply %3275, %3277 : tensor<1x8x16x64xbf16>
    %3279 = stablehlo.broadcast_in_dim %3274, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3280 = stablehlo.multiply %3276, %3279 : tensor<1x8x16x64xbf16>
    %3281 = stablehlo.subtract %3278, %3280 : tensor<1x8x16x64xbf16>
    %3282 = stablehlo.broadcast_in_dim %3272, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3283 = stablehlo.multiply %3276, %3282 : tensor<1x8x16x64xbf16>
    %3284 = stablehlo.broadcast_in_dim %3274, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3285 = stablehlo.multiply %3275, %3284 : tensor<1x8x16x64xbf16>
    %3286 = stablehlo.add %3283, %3285 : tensor<1x8x16x64xbf16>
    %3287 = stablehlo.concatenate %3281, %3286, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3288 = stablehlo.broadcast_in_dim %3269, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3289 = stablehlo.convert %3288 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3290 = stablehlo.broadcast_in_dim %3270, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3291 = stablehlo.convert %3290 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3292 = stablehlo.slice %3260 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3293 = stablehlo.slice %3260 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3294 = stablehlo.broadcast_in_dim %3289, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3295 = stablehlo.multiply %3292, %3294 : tensor<1x8x8x64xbf16>
    %3296 = stablehlo.broadcast_in_dim %3291, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3297 = stablehlo.multiply %3293, %3296 : tensor<1x8x8x64xbf16>
    %3298 = stablehlo.subtract %3295, %3297 : tensor<1x8x8x64xbf16>
    %3299 = stablehlo.broadcast_in_dim %3289, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3300 = stablehlo.multiply %3293, %3299 : tensor<1x8x8x64xbf16>
    %3301 = stablehlo.broadcast_in_dim %3291, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3302 = stablehlo.multiply %3292, %3301 : tensor<1x8x8x64xbf16>
    %3303 = stablehlo.add %3300, %3302 : tensor<1x8x8x64xbf16>
    %3304 = stablehlo.concatenate %3298, %3303, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3305 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3306 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3307 = stablehlo.broadcast_in_dim %3305, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3308 = stablehlo.and %3307, %3306 : tensor<1x1x8x8xi1>
    %3309 = stablehlo.convert %3308 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3310 = sdy.sharding_constraint %3287 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3311 = sdy.sharding_constraint %3304 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3312 = sdy.sharding_constraint %3261 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3313 = sdy.sharding_constraint %3309 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3314 = stablehlo.reshape %3310 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3315 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3316 = stablehlo.multiply %3314, %3315 : tensor<1x8x8x2x128xbf16>
    %3317 = stablehlo.dot_general %3311, %3316, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3318 = stablehlo.transpose %3317, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3319 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3320 = stablehlo.compare  NE, %3313, %3319,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3321 = stablehlo.convert %3320 : tensor<1x1x8x8xi1>
    %3322 = stablehlo.broadcast_in_dim %3321, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3323 = stablehlo.reshape %3322 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3324 = call @_where_91(%3323, %3318, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3325 = stablehlo.convert %3324 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_140 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3326 = stablehlo.reduce(%3325 init: %cst_140) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3327 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3328 = stablehlo.maximum %3327, %3326 : tensor<1x8x2x8xf32>
    %3329 = stablehlo.broadcast_in_dim %3328, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3330 = stablehlo.broadcast_in_dim %3329, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3331 = stablehlo.subtract %3325, %3330 : tensor<1x8x2x8x8xf32>
    %3332 = stablehlo.exponential %3331 : tensor<1x8x2x8x8xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3333 = stablehlo.reduce(%3332 init: %cst_141) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3334 = stablehlo.broadcast_in_dim %3333, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3335 = stablehlo.broadcast_in_dim %3334, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3336 = stablehlo.divide %3332, %3335 : tensor<1x8x2x8x8xf32>
    %3337 = stablehlo.convert %3336 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3338 = stablehlo.dot_general %3312, %3337, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3339 = stablehlo.transpose %3338, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3340 = stablehlo.reshape %3339 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3341 = sdy.sharding_constraint %3340 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3342 = stablehlo.reshape %3341 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3343 = stablehlo.convert %arg318 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3344 = stablehlo.dot_general %3342, %3343, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3345 = stablehlo.add %3194, %3344 : tensor<1x8x1024xbf16>
    %3346 = stablehlo.convert %3345 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3347 = stablehlo.multiply %3346, %3346 : tensor<1x8x1024xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3348 = stablehlo.reduce(%3347 init: %cst_142) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3349 = stablehlo.broadcast_in_dim %3348, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3350 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3351 = stablehlo.divide %3349, %3350 : tensor<1x8x1xf32>
    %3352 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3353 = stablehlo.add %3351, %3352 : tensor<1x8x1xf32>
    %3354 = stablehlo.rsqrt %3353 : tensor<1x8x1xf32>
    %3355 = stablehlo.broadcast_in_dim %3354, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3356 = stablehlo.multiply %3346, %3355 : tensor<1x8x1024xf32>
    %3357 = stablehlo.convert %3356 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3358 = stablehlo.convert %arg315 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3359 = stablehlo.broadcast_in_dim %3358, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3360 = stablehlo.broadcast_in_dim %3359, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3361 = stablehlo.multiply %3360, %3357 : tensor<1x8x1024xbf16>
    %3362 = stablehlo.convert %arg313 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3363 = stablehlo.dot_general %3361, %3362, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3364 = call @silu(%3363) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3365 = stablehlo.convert %arg314 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3366 = stablehlo.dot_general %3361, %3365, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3367 = stablehlo.multiply %3364, %3366 : tensor<1x8x3072xbf16>
    %3368 = stablehlo.convert %arg312 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3369 = stablehlo.dot_general %3367, %3368, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3370 = stablehlo.add %3345, %3369 : tensor<1x8x1024xbf16>
    %3371 = stablehlo.convert %3370 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3372 = stablehlo.multiply %3371, %3371 : tensor<1x8x1024xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3373 = stablehlo.reduce(%3372 init: %cst_143) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3374 = stablehlo.broadcast_in_dim %3373, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3375 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3376 = stablehlo.divide %3374, %3375 : tensor<1x8x1xf32>
    %3377 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3378 = stablehlo.add %3376, %3377 : tensor<1x8x1xf32>
    %3379 = stablehlo.rsqrt %3378 : tensor<1x8x1xf32>
    %3380 = stablehlo.broadcast_in_dim %3379, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3381 = stablehlo.multiply %3371, %3380 : tensor<1x8x1024xf32>
    %3382 = stablehlo.convert %3381 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3383 = stablehlo.convert %arg322 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3384 = stablehlo.broadcast_in_dim %3383, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3385 = stablehlo.broadcast_in_dim %3384, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3386 = stablehlo.multiply %3385, %3382 : tensor<1x8x1024xbf16>
    %3387 = stablehlo.convert %arg76 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3388 = stablehlo.convert %arg77 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3389 = stablehlo.dot_general %3386, %3387, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3390 = stablehlo.dot_general %3389, %3388, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3391 = stablehlo.convert %arg331 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3392 = stablehlo.dot_general %3386, %3391, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3393 = stablehlo.add %3390, %3392 : tensor<1x8x2048xbf16>
    %3394 = stablehlo.convert %arg328 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3395 = stablehlo.dot_general %3386, %3394, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3396 = stablehlo.convert %arg78 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3397 = stablehlo.convert %arg79 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3398 = stablehlo.dot_general %3386, %3396, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3399 = stablehlo.dot_general %3398, %3397, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3400 = stablehlo.convert %arg332 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3401 = stablehlo.dot_general %3386, %3400, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3402 = stablehlo.add %3399, %3401 : tensor<1x8x1024xbf16>
    %3403 = stablehlo.reshape %3393 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3404 = stablehlo.convert %3403 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3405 = stablehlo.multiply %3404, %3404 : tensor<1x8x16x128xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3406 = stablehlo.reduce(%3405 init: %cst_144) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3407 = stablehlo.broadcast_in_dim %3406, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3408 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3409 = stablehlo.divide %3407, %3408 : tensor<1x8x16x1xf32>
    %3410 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3411 = stablehlo.add %3409, %3410 : tensor<1x8x16x1xf32>
    %3412 = stablehlo.rsqrt %3411 : tensor<1x8x16x1xf32>
    %3413 = stablehlo.broadcast_in_dim %3412, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3414 = stablehlo.multiply %3404, %3413 : tensor<1x8x16x128xf32>
    %3415 = stablehlo.convert %3414 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3416 = stablehlo.convert %arg330 : (tensor<128xf32>) -> tensor<128xbf16>
    %3417 = stablehlo.broadcast_in_dim %3416, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3418 = stablehlo.broadcast_in_dim %3417, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3419 = stablehlo.multiply %3418, %3415 : tensor<1x8x16x128xbf16>
    %3420 = stablehlo.reshape %3395 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3421 = stablehlo.convert %3420 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3422 = stablehlo.multiply %3421, %3421 : tensor<1x8x8x128xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3423 = stablehlo.reduce(%3422 init: %cst_145) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3424 = stablehlo.broadcast_in_dim %3423, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3425 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3426 = stablehlo.divide %3424, %3425 : tensor<1x8x8x1xf32>
    %3427 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3428 = stablehlo.add %3426, %3427 : tensor<1x8x8x1xf32>
    %3429 = stablehlo.rsqrt %3428 : tensor<1x8x8x1xf32>
    %3430 = stablehlo.broadcast_in_dim %3429, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3431 = stablehlo.multiply %3421, %3430 : tensor<1x8x8x128xf32>
    %3432 = stablehlo.convert %3431 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3433 = stablehlo.convert %arg327 : (tensor<128xf32>) -> tensor<128xbf16>
    %3434 = stablehlo.broadcast_in_dim %3433, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3435 = stablehlo.broadcast_in_dim %3434, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3436 = stablehlo.multiply %3435, %3432 : tensor<1x8x8x128xbf16>
    %3437 = stablehlo.reshape %3402 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3438 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3439 = stablehlo.compare  LT, %7, %3438,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3440 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3441 = stablehlo.add %7, %3440 : tensor<1x8xi32>
    %3442 = stablehlo.select %3439, %3441, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3443 = stablehlo.broadcast_in_dim %3442, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3444 = "stablehlo.gather"(%26, %3443) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3445 = stablehlo.slice %3444 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3446 = stablehlo.slice %3444 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3447 = stablehlo.broadcast_in_dim %3445, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3448 = stablehlo.convert %3447 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3449 = stablehlo.broadcast_in_dim %3446, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3450 = stablehlo.convert %3449 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3451 = stablehlo.slice %3419 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3452 = stablehlo.slice %3419 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3453 = stablehlo.broadcast_in_dim %3448, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3454 = stablehlo.multiply %3451, %3453 : tensor<1x8x16x64xbf16>
    %3455 = stablehlo.broadcast_in_dim %3450, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3456 = stablehlo.multiply %3452, %3455 : tensor<1x8x16x64xbf16>
    %3457 = stablehlo.subtract %3454, %3456 : tensor<1x8x16x64xbf16>
    %3458 = stablehlo.broadcast_in_dim %3448, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3459 = stablehlo.multiply %3452, %3458 : tensor<1x8x16x64xbf16>
    %3460 = stablehlo.broadcast_in_dim %3450, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3461 = stablehlo.multiply %3451, %3460 : tensor<1x8x16x64xbf16>
    %3462 = stablehlo.add %3459, %3461 : tensor<1x8x16x64xbf16>
    %3463 = stablehlo.concatenate %3457, %3462, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3464 = stablehlo.broadcast_in_dim %3445, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3465 = stablehlo.convert %3464 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3466 = stablehlo.broadcast_in_dim %3446, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3467 = stablehlo.convert %3466 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3468 = stablehlo.slice %3436 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3469 = stablehlo.slice %3436 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3470 = stablehlo.broadcast_in_dim %3465, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3471 = stablehlo.multiply %3468, %3470 : tensor<1x8x8x64xbf16>
    %3472 = stablehlo.broadcast_in_dim %3467, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3473 = stablehlo.multiply %3469, %3472 : tensor<1x8x8x64xbf16>
    %3474 = stablehlo.subtract %3471, %3473 : tensor<1x8x8x64xbf16>
    %3475 = stablehlo.broadcast_in_dim %3465, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3476 = stablehlo.multiply %3469, %3475 : tensor<1x8x8x64xbf16>
    %3477 = stablehlo.broadcast_in_dim %3467, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3478 = stablehlo.multiply %3468, %3477 : tensor<1x8x8x64xbf16>
    %3479 = stablehlo.add %3476, %3478 : tensor<1x8x8x64xbf16>
    %3480 = stablehlo.concatenate %3474, %3479, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3481 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3482 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3483 = stablehlo.broadcast_in_dim %3481, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3484 = stablehlo.and %3483, %3482 : tensor<1x1x8x8xi1>
    %3485 = stablehlo.convert %3484 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3486 = sdy.sharding_constraint %3463 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3487 = sdy.sharding_constraint %3480 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3488 = sdy.sharding_constraint %3437 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3489 = sdy.sharding_constraint %3485 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3490 = stablehlo.reshape %3486 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3491 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3492 = stablehlo.multiply %3490, %3491 : tensor<1x8x8x2x128xbf16>
    %3493 = stablehlo.dot_general %3487, %3492, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3494 = stablehlo.transpose %3493, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3495 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3496 = stablehlo.compare  NE, %3489, %3495,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3497 = stablehlo.convert %3496 : tensor<1x1x8x8xi1>
    %3498 = stablehlo.broadcast_in_dim %3497, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3499 = stablehlo.reshape %3498 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3500 = call @_where_91(%3499, %3494, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3501 = stablehlo.convert %3500 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_147 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3502 = stablehlo.reduce(%3501 init: %cst_147) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3503 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3504 = stablehlo.maximum %3503, %3502 : tensor<1x8x2x8xf32>
    %3505 = stablehlo.broadcast_in_dim %3504, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3506 = stablehlo.broadcast_in_dim %3505, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3507 = stablehlo.subtract %3501, %3506 : tensor<1x8x2x8x8xf32>
    %3508 = stablehlo.exponential %3507 : tensor<1x8x2x8x8xf32>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3509 = stablehlo.reduce(%3508 init: %cst_148) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3510 = stablehlo.broadcast_in_dim %3509, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3511 = stablehlo.broadcast_in_dim %3510, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3512 = stablehlo.divide %3508, %3511 : tensor<1x8x2x8x8xf32>
    %3513 = stablehlo.convert %3512 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3514 = stablehlo.dot_general %3488, %3513, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3515 = stablehlo.transpose %3514, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3516 = stablehlo.reshape %3515 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3517 = sdy.sharding_constraint %3516 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3518 = stablehlo.reshape %3517 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3519 = stablehlo.convert %arg329 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3520 = stablehlo.dot_general %3518, %3519, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3521 = stablehlo.add %3370, %3520 : tensor<1x8x1024xbf16>
    %3522 = stablehlo.convert %3521 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3523 = stablehlo.multiply %3522, %3522 : tensor<1x8x1024xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3524 = stablehlo.reduce(%3523 init: %cst_149) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3525 = stablehlo.broadcast_in_dim %3524, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3526 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3527 = stablehlo.divide %3525, %3526 : tensor<1x8x1xf32>
    %3528 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3529 = stablehlo.add %3527, %3528 : tensor<1x8x1xf32>
    %3530 = stablehlo.rsqrt %3529 : tensor<1x8x1xf32>
    %3531 = stablehlo.broadcast_in_dim %3530, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3532 = stablehlo.multiply %3522, %3531 : tensor<1x8x1024xf32>
    %3533 = stablehlo.convert %3532 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3534 = stablehlo.convert %arg326 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3535 = stablehlo.broadcast_in_dim %3534, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3536 = stablehlo.broadcast_in_dim %3535, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3537 = stablehlo.multiply %3536, %3533 : tensor<1x8x1024xbf16>
    %3538 = stablehlo.convert %arg324 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3539 = stablehlo.dot_general %3537, %3538, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3540 = call @silu(%3539) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3541 = stablehlo.convert %arg325 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3542 = stablehlo.dot_general %3537, %3541, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3543 = stablehlo.multiply %3540, %3542 : tensor<1x8x3072xbf16>
    %3544 = stablehlo.convert %arg323 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3545 = stablehlo.dot_general %3543, %3544, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3546 = stablehlo.add %3521, %3545 : tensor<1x8x1024xbf16>
    %3547 = stablehlo.convert %3546 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3548 = stablehlo.multiply %3547, %3547 : tensor<1x8x1024xf32>
    %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3549 = stablehlo.reduce(%3548 init: %cst_150) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3550 = stablehlo.broadcast_in_dim %3549, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3551 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3552 = stablehlo.divide %3550, %3551 : tensor<1x8x1xf32>
    %3553 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3554 = stablehlo.add %3552, %3553 : tensor<1x8x1xf32>
    %3555 = stablehlo.rsqrt %3554 : tensor<1x8x1xf32>
    %3556 = stablehlo.broadcast_in_dim %3555, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3557 = stablehlo.multiply %3547, %3556 : tensor<1x8x1024xf32>
    %3558 = stablehlo.convert %3557 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3559 = stablehlo.convert %arg333 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3560 = stablehlo.broadcast_in_dim %3559, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3561 = stablehlo.broadcast_in_dim %3560, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3562 = stablehlo.multiply %3561, %3558 : tensor<1x8x1024xbf16>
    %3563 = stablehlo.convert %arg80 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3564 = stablehlo.convert %arg81 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3565 = stablehlo.dot_general %3562, %3563, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3566 = stablehlo.dot_general %3565, %3564, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3567 = stablehlo.convert %arg342 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3568 = stablehlo.dot_general %3562, %3567, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3569 = stablehlo.add %3566, %3568 : tensor<1x8x2048xbf16>
    %3570 = stablehlo.convert %arg339 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3571 = stablehlo.dot_general %3562, %3570, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3572 = stablehlo.convert %arg82 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3573 = stablehlo.convert %arg83 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3574 = stablehlo.dot_general %3562, %3572, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3575 = stablehlo.dot_general %3574, %3573, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3576 = stablehlo.convert %arg343 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3577 = stablehlo.dot_general %3562, %3576, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3578 = stablehlo.add %3575, %3577 : tensor<1x8x1024xbf16>
    %3579 = stablehlo.reshape %3569 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3580 = stablehlo.convert %3579 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3581 = stablehlo.multiply %3580, %3580 : tensor<1x8x16x128xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3582 = stablehlo.reduce(%3581 init: %cst_151) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3583 = stablehlo.broadcast_in_dim %3582, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3584 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3585 = stablehlo.divide %3583, %3584 : tensor<1x8x16x1xf32>
    %3586 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3587 = stablehlo.add %3585, %3586 : tensor<1x8x16x1xf32>
    %3588 = stablehlo.rsqrt %3587 : tensor<1x8x16x1xf32>
    %3589 = stablehlo.broadcast_in_dim %3588, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3590 = stablehlo.multiply %3580, %3589 : tensor<1x8x16x128xf32>
    %3591 = stablehlo.convert %3590 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3592 = stablehlo.convert %arg341 : (tensor<128xf32>) -> tensor<128xbf16>
    %3593 = stablehlo.broadcast_in_dim %3592, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3594 = stablehlo.broadcast_in_dim %3593, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3595 = stablehlo.multiply %3594, %3591 : tensor<1x8x16x128xbf16>
    %3596 = stablehlo.reshape %3571 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3597 = stablehlo.convert %3596 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3598 = stablehlo.multiply %3597, %3597 : tensor<1x8x8x128xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3599 = stablehlo.reduce(%3598 init: %cst_152) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3600 = stablehlo.broadcast_in_dim %3599, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3601 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3602 = stablehlo.divide %3600, %3601 : tensor<1x8x8x1xf32>
    %3603 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3604 = stablehlo.add %3602, %3603 : tensor<1x8x8x1xf32>
    %3605 = stablehlo.rsqrt %3604 : tensor<1x8x8x1xf32>
    %3606 = stablehlo.broadcast_in_dim %3605, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3607 = stablehlo.multiply %3597, %3606 : tensor<1x8x8x128xf32>
    %3608 = stablehlo.convert %3607 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3609 = stablehlo.convert %arg338 : (tensor<128xf32>) -> tensor<128xbf16>
    %3610 = stablehlo.broadcast_in_dim %3609, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3611 = stablehlo.broadcast_in_dim %3610, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3612 = stablehlo.multiply %3611, %3608 : tensor<1x8x8x128xbf16>
    %3613 = stablehlo.reshape %3578 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3614 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3615 = stablehlo.compare  LT, %7, %3614,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3616 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3617 = stablehlo.add %7, %3616 : tensor<1x8xi32>
    %3618 = stablehlo.select %3615, %3617, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3619 = stablehlo.broadcast_in_dim %3618, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3620 = "stablehlo.gather"(%26, %3619) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3621 = stablehlo.slice %3620 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3622 = stablehlo.slice %3620 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3623 = stablehlo.broadcast_in_dim %3621, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3624 = stablehlo.convert %3623 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3625 = stablehlo.broadcast_in_dim %3622, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3626 = stablehlo.convert %3625 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3627 = stablehlo.slice %3595 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3628 = stablehlo.slice %3595 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3629 = stablehlo.broadcast_in_dim %3624, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3630 = stablehlo.multiply %3627, %3629 : tensor<1x8x16x64xbf16>
    %3631 = stablehlo.broadcast_in_dim %3626, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3632 = stablehlo.multiply %3628, %3631 : tensor<1x8x16x64xbf16>
    %3633 = stablehlo.subtract %3630, %3632 : tensor<1x8x16x64xbf16>
    %3634 = stablehlo.broadcast_in_dim %3624, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3635 = stablehlo.multiply %3628, %3634 : tensor<1x8x16x64xbf16>
    %3636 = stablehlo.broadcast_in_dim %3626, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3637 = stablehlo.multiply %3627, %3636 : tensor<1x8x16x64xbf16>
    %3638 = stablehlo.add %3635, %3637 : tensor<1x8x16x64xbf16>
    %3639 = stablehlo.concatenate %3633, %3638, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3640 = stablehlo.broadcast_in_dim %3621, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3641 = stablehlo.convert %3640 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3642 = stablehlo.broadcast_in_dim %3622, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3643 = stablehlo.convert %3642 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3644 = stablehlo.slice %3612 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3645 = stablehlo.slice %3612 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3646 = stablehlo.broadcast_in_dim %3641, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3647 = stablehlo.multiply %3644, %3646 : tensor<1x8x8x64xbf16>
    %3648 = stablehlo.broadcast_in_dim %3643, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3649 = stablehlo.multiply %3645, %3648 : tensor<1x8x8x64xbf16>
    %3650 = stablehlo.subtract %3647, %3649 : tensor<1x8x8x64xbf16>
    %3651 = stablehlo.broadcast_in_dim %3641, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3652 = stablehlo.multiply %3645, %3651 : tensor<1x8x8x64xbf16>
    %3653 = stablehlo.broadcast_in_dim %3643, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3654 = stablehlo.multiply %3644, %3653 : tensor<1x8x8x64xbf16>
    %3655 = stablehlo.add %3652, %3654 : tensor<1x8x8x64xbf16>
    %3656 = stablehlo.concatenate %3650, %3655, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3657 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3658 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3659 = stablehlo.broadcast_in_dim %3657, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3660 = stablehlo.and %3659, %3658 : tensor<1x1x8x8xi1>
    %3661 = stablehlo.convert %3660 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3662 = sdy.sharding_constraint %3639 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3663 = sdy.sharding_constraint %3656 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3664 = sdy.sharding_constraint %3613 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3665 = sdy.sharding_constraint %3661 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3666 = stablehlo.reshape %3662 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3667 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3668 = stablehlo.multiply %3666, %3667 : tensor<1x8x8x2x128xbf16>
    %3669 = stablehlo.dot_general %3663, %3668, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3670 = stablehlo.transpose %3669, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3671 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3672 = stablehlo.compare  NE, %3665, %3671,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3673 = stablehlo.convert %3672 : tensor<1x1x8x8xi1>
    %3674 = stablehlo.broadcast_in_dim %3673, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3675 = stablehlo.reshape %3674 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3676 = call @_where_91(%3675, %3670, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3677 = stablehlo.convert %3676 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_154 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3678 = stablehlo.reduce(%3677 init: %cst_154) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3679 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3680 = stablehlo.maximum %3679, %3678 : tensor<1x8x2x8xf32>
    %3681 = stablehlo.broadcast_in_dim %3680, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3682 = stablehlo.broadcast_in_dim %3681, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3683 = stablehlo.subtract %3677, %3682 : tensor<1x8x2x8x8xf32>
    %3684 = stablehlo.exponential %3683 : tensor<1x8x2x8x8xf32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3685 = stablehlo.reduce(%3684 init: %cst_155) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3686 = stablehlo.broadcast_in_dim %3685, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3687 = stablehlo.broadcast_in_dim %3686, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3688 = stablehlo.divide %3684, %3687 : tensor<1x8x2x8x8xf32>
    %3689 = stablehlo.convert %3688 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3690 = stablehlo.dot_general %3664, %3689, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3691 = stablehlo.transpose %3690, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3692 = stablehlo.reshape %3691 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3693 = sdy.sharding_constraint %3692 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3694 = stablehlo.reshape %3693 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3695 = stablehlo.convert %arg340 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3696 = stablehlo.dot_general %3694, %3695, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3697 = stablehlo.add %3546, %3696 : tensor<1x8x1024xbf16>
    %3698 = stablehlo.convert %3697 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3699 = stablehlo.multiply %3698, %3698 : tensor<1x8x1024xf32>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3700 = stablehlo.reduce(%3699 init: %cst_156) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3701 = stablehlo.broadcast_in_dim %3700, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3702 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3703 = stablehlo.divide %3701, %3702 : tensor<1x8x1xf32>
    %3704 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3705 = stablehlo.add %3703, %3704 : tensor<1x8x1xf32>
    %3706 = stablehlo.rsqrt %3705 : tensor<1x8x1xf32>
    %3707 = stablehlo.broadcast_in_dim %3706, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3708 = stablehlo.multiply %3698, %3707 : tensor<1x8x1024xf32>
    %3709 = stablehlo.convert %3708 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3710 = stablehlo.convert %arg337 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3711 = stablehlo.broadcast_in_dim %3710, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3712 = stablehlo.broadcast_in_dim %3711, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3713 = stablehlo.multiply %3712, %3709 : tensor<1x8x1024xbf16>
    %3714 = stablehlo.convert %arg335 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3715 = stablehlo.dot_general %3713, %3714, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3716 = call @silu(%3715) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3717 = stablehlo.convert %arg336 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3718 = stablehlo.dot_general %3713, %3717, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3719 = stablehlo.multiply %3716, %3718 : tensor<1x8x3072xbf16>
    %3720 = stablehlo.convert %arg334 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3721 = stablehlo.dot_general %3719, %3720, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3722 = stablehlo.add %3697, %3721 : tensor<1x8x1024xbf16>
    %3723 = stablehlo.convert %3722 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3724 = stablehlo.multiply %3723, %3723 : tensor<1x8x1024xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3725 = stablehlo.reduce(%3724 init: %cst_157) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3726 = stablehlo.broadcast_in_dim %3725, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3727 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3728 = stablehlo.divide %3726, %3727 : tensor<1x8x1xf32>
    %3729 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3730 = stablehlo.add %3728, %3729 : tensor<1x8x1xf32>
    %3731 = stablehlo.rsqrt %3730 : tensor<1x8x1xf32>
    %3732 = stablehlo.broadcast_in_dim %3731, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3733 = stablehlo.multiply %3723, %3732 : tensor<1x8x1024xf32>
    %3734 = stablehlo.convert %3733 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3735 = stablehlo.convert %arg344 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3736 = stablehlo.broadcast_in_dim %3735, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3737 = stablehlo.broadcast_in_dim %3736, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3738 = stablehlo.multiply %3737, %3734 : tensor<1x8x1024xbf16>
    %3739 = stablehlo.convert %arg84 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3740 = stablehlo.convert %arg85 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3741 = stablehlo.dot_general %3738, %3739, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3742 = stablehlo.dot_general %3741, %3740, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3743 = stablehlo.convert %arg353 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3744 = stablehlo.dot_general %3738, %3743, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3745 = stablehlo.add %3742, %3744 : tensor<1x8x2048xbf16>
    %3746 = stablehlo.convert %arg350 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3747 = stablehlo.dot_general %3738, %3746, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3748 = stablehlo.convert %arg86 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3749 = stablehlo.convert %arg87 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3750 = stablehlo.dot_general %3738, %3748, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3751 = stablehlo.dot_general %3750, %3749, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3752 = stablehlo.convert %arg354 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3753 = stablehlo.dot_general %3738, %3752, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3754 = stablehlo.add %3751, %3753 : tensor<1x8x1024xbf16>
    %3755 = stablehlo.reshape %3745 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3756 = stablehlo.convert %3755 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3757 = stablehlo.multiply %3756, %3756 : tensor<1x8x16x128xf32>
    %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3758 = stablehlo.reduce(%3757 init: %cst_158) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3759 = stablehlo.broadcast_in_dim %3758, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3760 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3761 = stablehlo.divide %3759, %3760 : tensor<1x8x16x1xf32>
    %3762 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3763 = stablehlo.add %3761, %3762 : tensor<1x8x16x1xf32>
    %3764 = stablehlo.rsqrt %3763 : tensor<1x8x16x1xf32>
    %3765 = stablehlo.broadcast_in_dim %3764, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3766 = stablehlo.multiply %3756, %3765 : tensor<1x8x16x128xf32>
    %3767 = stablehlo.convert %3766 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3768 = stablehlo.convert %arg352 : (tensor<128xf32>) -> tensor<128xbf16>
    %3769 = stablehlo.broadcast_in_dim %3768, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3770 = stablehlo.broadcast_in_dim %3769, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3771 = stablehlo.multiply %3770, %3767 : tensor<1x8x16x128xbf16>
    %3772 = stablehlo.reshape %3747 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3773 = stablehlo.convert %3772 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3774 = stablehlo.multiply %3773, %3773 : tensor<1x8x8x128xf32>
    %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3775 = stablehlo.reduce(%3774 init: %cst_159) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3776 = stablehlo.broadcast_in_dim %3775, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3777 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3778 = stablehlo.divide %3776, %3777 : tensor<1x8x8x1xf32>
    %3779 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3780 = stablehlo.add %3778, %3779 : tensor<1x8x8x1xf32>
    %3781 = stablehlo.rsqrt %3780 : tensor<1x8x8x1xf32>
    %3782 = stablehlo.broadcast_in_dim %3781, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3783 = stablehlo.multiply %3773, %3782 : tensor<1x8x8x128xf32>
    %3784 = stablehlo.convert %3783 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3785 = stablehlo.convert %arg349 : (tensor<128xf32>) -> tensor<128xbf16>
    %3786 = stablehlo.broadcast_in_dim %3785, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3787 = stablehlo.broadcast_in_dim %3786, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3788 = stablehlo.multiply %3787, %3784 : tensor<1x8x8x128xbf16>
    %3789 = stablehlo.reshape %3754 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3790 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3791 = stablehlo.compare  LT, %7, %3790,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3792 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3793 = stablehlo.add %7, %3792 : tensor<1x8xi32>
    %3794 = stablehlo.select %3791, %3793, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3795 = stablehlo.broadcast_in_dim %3794, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3796 = "stablehlo.gather"(%26, %3795) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3797 = stablehlo.slice %3796 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3798 = stablehlo.slice %3796 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3799 = stablehlo.broadcast_in_dim %3797, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3800 = stablehlo.convert %3799 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3801 = stablehlo.broadcast_in_dim %3798, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3802 = stablehlo.convert %3801 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3803 = stablehlo.slice %3771 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3804 = stablehlo.slice %3771 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3805 = stablehlo.broadcast_in_dim %3800, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3806 = stablehlo.multiply %3803, %3805 : tensor<1x8x16x64xbf16>
    %3807 = stablehlo.broadcast_in_dim %3802, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3808 = stablehlo.multiply %3804, %3807 : tensor<1x8x16x64xbf16>
    %3809 = stablehlo.subtract %3806, %3808 : tensor<1x8x16x64xbf16>
    %3810 = stablehlo.broadcast_in_dim %3800, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3811 = stablehlo.multiply %3804, %3810 : tensor<1x8x16x64xbf16>
    %3812 = stablehlo.broadcast_in_dim %3802, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3813 = stablehlo.multiply %3803, %3812 : tensor<1x8x16x64xbf16>
    %3814 = stablehlo.add %3811, %3813 : tensor<1x8x16x64xbf16>
    %3815 = stablehlo.concatenate %3809, %3814, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3816 = stablehlo.broadcast_in_dim %3797, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3817 = stablehlo.convert %3816 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3818 = stablehlo.broadcast_in_dim %3798, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3819 = stablehlo.convert %3818 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3820 = stablehlo.slice %3788 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3821 = stablehlo.slice %3788 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3822 = stablehlo.broadcast_in_dim %3817, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3823 = stablehlo.multiply %3820, %3822 : tensor<1x8x8x64xbf16>
    %3824 = stablehlo.broadcast_in_dim %3819, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3825 = stablehlo.multiply %3821, %3824 : tensor<1x8x8x64xbf16>
    %3826 = stablehlo.subtract %3823, %3825 : tensor<1x8x8x64xbf16>
    %3827 = stablehlo.broadcast_in_dim %3817, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3828 = stablehlo.multiply %3821, %3827 : tensor<1x8x8x64xbf16>
    %3829 = stablehlo.broadcast_in_dim %3819, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3830 = stablehlo.multiply %3820, %3829 : tensor<1x8x8x64xbf16>
    %3831 = stablehlo.add %3828, %3830 : tensor<1x8x8x64xbf16>
    %3832 = stablehlo.concatenate %3826, %3831, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3833 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3834 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3835 = stablehlo.broadcast_in_dim %3833, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3836 = stablehlo.and %3835, %3834 : tensor<1x1x8x8xi1>
    %3837 = stablehlo.convert %3836 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3838 = sdy.sharding_constraint %3815 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3839 = sdy.sharding_constraint %3832 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3840 = sdy.sharding_constraint %3789 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3841 = sdy.sharding_constraint %3837 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3842 = stablehlo.reshape %3838 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3843 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3844 = stablehlo.multiply %3842, %3843 : tensor<1x8x8x2x128xbf16>
    %3845 = stablehlo.dot_general %3839, %3844, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3846 = stablehlo.transpose %3845, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3847 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3848 = stablehlo.compare  NE, %3841, %3847,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3849 = stablehlo.convert %3848 : tensor<1x1x8x8xi1>
    %3850 = stablehlo.broadcast_in_dim %3849, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3851 = stablehlo.reshape %3850 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3852 = call @_where_91(%3851, %3846, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3853 = stablehlo.convert %3852 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_161 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3854 = stablehlo.reduce(%3853 init: %cst_161) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3855 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3856 = stablehlo.maximum %3855, %3854 : tensor<1x8x2x8xf32>
    %3857 = stablehlo.broadcast_in_dim %3856, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3858 = stablehlo.broadcast_in_dim %3857, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3859 = stablehlo.subtract %3853, %3858 : tensor<1x8x2x8x8xf32>
    %3860 = stablehlo.exponential %3859 : tensor<1x8x2x8x8xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3861 = stablehlo.reduce(%3860 init: %cst_162) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3862 = stablehlo.broadcast_in_dim %3861, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3863 = stablehlo.broadcast_in_dim %3862, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3864 = stablehlo.divide %3860, %3863 : tensor<1x8x2x8x8xf32>
    %3865 = stablehlo.convert %3864 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3866 = stablehlo.dot_general %3840, %3865, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3867 = stablehlo.transpose %3866, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3868 = stablehlo.reshape %3867 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3869 = sdy.sharding_constraint %3868 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3870 = stablehlo.reshape %3869 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3871 = stablehlo.convert %arg351 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3872 = stablehlo.dot_general %3870, %3871, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3873 = stablehlo.add %3722, %3872 : tensor<1x8x1024xbf16>
    %3874 = stablehlo.convert %3873 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3875 = stablehlo.multiply %3874, %3874 : tensor<1x8x1024xf32>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3876 = stablehlo.reduce(%3875 init: %cst_163) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3877 = stablehlo.broadcast_in_dim %3876, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3878 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3879 = stablehlo.divide %3877, %3878 : tensor<1x8x1xf32>
    %3880 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3881 = stablehlo.add %3879, %3880 : tensor<1x8x1xf32>
    %3882 = stablehlo.rsqrt %3881 : tensor<1x8x1xf32>
    %3883 = stablehlo.broadcast_in_dim %3882, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3884 = stablehlo.multiply %3874, %3883 : tensor<1x8x1024xf32>
    %3885 = stablehlo.convert %3884 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3886 = stablehlo.convert %arg348 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3887 = stablehlo.broadcast_in_dim %3886, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3888 = stablehlo.broadcast_in_dim %3887, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3889 = stablehlo.multiply %3888, %3885 : tensor<1x8x1024xbf16>
    %3890 = stablehlo.convert %arg346 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3891 = stablehlo.dot_general %3889, %3890, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3892 = call @silu(%3891) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3893 = stablehlo.convert %arg347 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3894 = stablehlo.dot_general %3889, %3893, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3895 = stablehlo.multiply %3892, %3894 : tensor<1x8x3072xbf16>
    %3896 = stablehlo.convert %arg345 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3897 = stablehlo.dot_general %3895, %3896, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3898 = stablehlo.add %3873, %3897 : tensor<1x8x1024xbf16>
    %3899 = stablehlo.convert %3898 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3900 = stablehlo.multiply %3899, %3899 : tensor<1x8x1024xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3901 = stablehlo.reduce(%3900 init: %cst_164) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3902 = stablehlo.broadcast_in_dim %3901, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3903 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3904 = stablehlo.divide %3902, %3903 : tensor<1x8x1xf32>
    %3905 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3906 = stablehlo.add %3904, %3905 : tensor<1x8x1xf32>
    %3907 = stablehlo.rsqrt %3906 : tensor<1x8x1xf32>
    %3908 = stablehlo.broadcast_in_dim %3907, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3909 = stablehlo.multiply %3899, %3908 : tensor<1x8x1024xf32>
    %3910 = stablehlo.convert %3909 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3911 = stablehlo.convert %arg355 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3912 = stablehlo.broadcast_in_dim %3911, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3913 = stablehlo.broadcast_in_dim %3912, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3914 = stablehlo.multiply %3913, %3910 : tensor<1x8x1024xbf16>
    %3915 = stablehlo.convert %arg88 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3916 = stablehlo.convert %arg89 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %3917 = stablehlo.dot_general %3914, %3915, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3918 = stablehlo.dot_general %3917, %3916, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3919 = stablehlo.convert %arg364 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3920 = stablehlo.dot_general %3914, %3919, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3921 = stablehlo.add %3918, %3920 : tensor<1x8x2048xbf16>
    %3922 = stablehlo.convert %arg361 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3923 = stablehlo.dot_general %3914, %3922, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3924 = stablehlo.convert %arg90 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %3925 = stablehlo.convert %arg91 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %3926 = stablehlo.dot_general %3914, %3924, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %3927 = stablehlo.dot_general %3926, %3925, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3928 = stablehlo.convert %arg365 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3929 = stablehlo.dot_general %3914, %3928, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3930 = stablehlo.add %3927, %3929 : tensor<1x8x1024xbf16>
    %3931 = stablehlo.reshape %3921 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3932 = stablehlo.convert %3931 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3933 = stablehlo.multiply %3932, %3932 : tensor<1x8x16x128xf32>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3934 = stablehlo.reduce(%3933 init: %cst_165) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3935 = stablehlo.broadcast_in_dim %3934, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3936 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3937 = stablehlo.divide %3935, %3936 : tensor<1x8x16x1xf32>
    %3938 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3939 = stablehlo.add %3937, %3938 : tensor<1x8x16x1xf32>
    %3940 = stablehlo.rsqrt %3939 : tensor<1x8x16x1xf32>
    %3941 = stablehlo.broadcast_in_dim %3940, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3942 = stablehlo.multiply %3932, %3941 : tensor<1x8x16x128xf32>
    %3943 = stablehlo.convert %3942 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3944 = stablehlo.convert %arg363 : (tensor<128xf32>) -> tensor<128xbf16>
    %3945 = stablehlo.broadcast_in_dim %3944, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3946 = stablehlo.broadcast_in_dim %3945, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3947 = stablehlo.multiply %3946, %3943 : tensor<1x8x16x128xbf16>
    %3948 = stablehlo.reshape %3923 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3949 = stablehlo.convert %3948 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3950 = stablehlo.multiply %3949, %3949 : tensor<1x8x8x128xf32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3951 = stablehlo.reduce(%3950 init: %cst_166) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3952 = stablehlo.broadcast_in_dim %3951, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3953 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3954 = stablehlo.divide %3952, %3953 : tensor<1x8x8x1xf32>
    %3955 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3956 = stablehlo.add %3954, %3955 : tensor<1x8x8x1xf32>
    %3957 = stablehlo.rsqrt %3956 : tensor<1x8x8x1xf32>
    %3958 = stablehlo.broadcast_in_dim %3957, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3959 = stablehlo.multiply %3949, %3958 : tensor<1x8x8x128xf32>
    %3960 = stablehlo.convert %3959 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3961 = stablehlo.convert %arg360 : (tensor<128xf32>) -> tensor<128xbf16>
    %3962 = stablehlo.broadcast_in_dim %3961, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3963 = stablehlo.broadcast_in_dim %3962, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3964 = stablehlo.multiply %3963, %3960 : tensor<1x8x8x128xbf16>
    %3965 = stablehlo.reshape %3930 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3966 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3967 = stablehlo.compare  LT, %7, %3966,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3968 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3969 = stablehlo.add %7, %3968 : tensor<1x8xi32>
    %3970 = stablehlo.select %3967, %3969, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3971 = stablehlo.broadcast_in_dim %3970, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3972 = "stablehlo.gather"(%26, %3971) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3973 = stablehlo.slice %3972 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3974 = stablehlo.slice %3972 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3975 = stablehlo.broadcast_in_dim %3973, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3976 = stablehlo.convert %3975 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3977 = stablehlo.broadcast_in_dim %3974, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3978 = stablehlo.convert %3977 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3979 = stablehlo.slice %3947 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3980 = stablehlo.slice %3947 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3981 = stablehlo.broadcast_in_dim %3976, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3982 = stablehlo.multiply %3979, %3981 : tensor<1x8x16x64xbf16>
    %3983 = stablehlo.broadcast_in_dim %3978, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3984 = stablehlo.multiply %3980, %3983 : tensor<1x8x16x64xbf16>
    %3985 = stablehlo.subtract %3982, %3984 : tensor<1x8x16x64xbf16>
    %3986 = stablehlo.broadcast_in_dim %3976, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3987 = stablehlo.multiply %3980, %3986 : tensor<1x8x16x64xbf16>
    %3988 = stablehlo.broadcast_in_dim %3978, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3989 = stablehlo.multiply %3979, %3988 : tensor<1x8x16x64xbf16>
    %3990 = stablehlo.add %3987, %3989 : tensor<1x8x16x64xbf16>
    %3991 = stablehlo.concatenate %3985, %3990, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3992 = stablehlo.broadcast_in_dim %3973, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3993 = stablehlo.convert %3992 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3994 = stablehlo.broadcast_in_dim %3974, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3995 = stablehlo.convert %3994 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3996 = stablehlo.slice %3964 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3997 = stablehlo.slice %3964 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3998 = stablehlo.broadcast_in_dim %3993, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3999 = stablehlo.multiply %3996, %3998 : tensor<1x8x8x64xbf16>
    %4000 = stablehlo.broadcast_in_dim %3995, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4001 = stablehlo.multiply %3997, %4000 : tensor<1x8x8x64xbf16>
    %4002 = stablehlo.subtract %3999, %4001 : tensor<1x8x8x64xbf16>
    %4003 = stablehlo.broadcast_in_dim %3993, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4004 = stablehlo.multiply %3997, %4003 : tensor<1x8x8x64xbf16>
    %4005 = stablehlo.broadcast_in_dim %3995, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4006 = stablehlo.multiply %3996, %4005 : tensor<1x8x8x64xbf16>
    %4007 = stablehlo.add %4004, %4006 : tensor<1x8x8x64xbf16>
    %4008 = stablehlo.concatenate %4002, %4007, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4009 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4010 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4011 = stablehlo.broadcast_in_dim %4009, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4012 = stablehlo.and %4011, %4010 : tensor<1x1x8x8xi1>
    %4013 = stablehlo.convert %4012 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4014 = sdy.sharding_constraint %3991 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4015 = sdy.sharding_constraint %4008 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4016 = sdy.sharding_constraint %3965 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4017 = sdy.sharding_constraint %4013 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4018 = stablehlo.reshape %4014 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4019 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4020 = stablehlo.multiply %4018, %4019 : tensor<1x8x8x2x128xbf16>
    %4021 = stablehlo.dot_general %4015, %4020, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4022 = stablehlo.transpose %4021, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4023 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4024 = stablehlo.compare  NE, %4017, %4023,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4025 = stablehlo.convert %4024 : tensor<1x1x8x8xi1>
    %4026 = stablehlo.broadcast_in_dim %4025, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4027 = stablehlo.reshape %4026 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4028 = call @_where_91(%4027, %4022, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4029 = stablehlo.convert %4028 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_168 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4030 = stablehlo.reduce(%4029 init: %cst_168) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4031 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4032 = stablehlo.maximum %4031, %4030 : tensor<1x8x2x8xf32>
    %4033 = stablehlo.broadcast_in_dim %4032, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4034 = stablehlo.broadcast_in_dim %4033, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4035 = stablehlo.subtract %4029, %4034 : tensor<1x8x2x8x8xf32>
    %4036 = stablehlo.exponential %4035 : tensor<1x8x2x8x8xf32>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4037 = stablehlo.reduce(%4036 init: %cst_169) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4038 = stablehlo.broadcast_in_dim %4037, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4039 = stablehlo.broadcast_in_dim %4038, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4040 = stablehlo.divide %4036, %4039 : tensor<1x8x2x8x8xf32>
    %4041 = stablehlo.convert %4040 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4042 = stablehlo.dot_general %4016, %4041, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4043 = stablehlo.transpose %4042, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4044 = stablehlo.reshape %4043 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4045 = sdy.sharding_constraint %4044 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4046 = stablehlo.reshape %4045 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4047 = stablehlo.convert %arg362 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4048 = stablehlo.dot_general %4046, %4047, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4049 = stablehlo.add %3898, %4048 : tensor<1x8x1024xbf16>
    %4050 = stablehlo.convert %4049 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4051 = stablehlo.multiply %4050, %4050 : tensor<1x8x1024xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4052 = stablehlo.reduce(%4051 init: %cst_170) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4053 = stablehlo.broadcast_in_dim %4052, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4054 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4055 = stablehlo.divide %4053, %4054 : tensor<1x8x1xf32>
    %4056 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4057 = stablehlo.add %4055, %4056 : tensor<1x8x1xf32>
    %4058 = stablehlo.rsqrt %4057 : tensor<1x8x1xf32>
    %4059 = stablehlo.broadcast_in_dim %4058, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4060 = stablehlo.multiply %4050, %4059 : tensor<1x8x1024xf32>
    %4061 = stablehlo.convert %4060 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4062 = stablehlo.convert %arg359 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4063 = stablehlo.broadcast_in_dim %4062, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4064 = stablehlo.broadcast_in_dim %4063, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4065 = stablehlo.multiply %4064, %4061 : tensor<1x8x1024xbf16>
    %4066 = stablehlo.convert %arg357 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4067 = stablehlo.dot_general %4065, %4066, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4068 = call @silu(%4067) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4069 = stablehlo.convert %arg358 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4070 = stablehlo.dot_general %4065, %4069, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4071 = stablehlo.multiply %4068, %4070 : tensor<1x8x3072xbf16>
    %4072 = stablehlo.convert %arg356 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4073 = stablehlo.dot_general %4071, %4072, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4074 = stablehlo.add %4049, %4073 : tensor<1x8x1024xbf16>
    %4075 = stablehlo.convert %4074 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4076 = stablehlo.multiply %4075, %4075 : tensor<1x8x1024xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4077 = stablehlo.reduce(%4076 init: %cst_171) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4078 = stablehlo.broadcast_in_dim %4077, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4079 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4080 = stablehlo.divide %4078, %4079 : tensor<1x8x1xf32>
    %4081 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4082 = stablehlo.add %4080, %4081 : tensor<1x8x1xf32>
    %4083 = stablehlo.rsqrt %4082 : tensor<1x8x1xf32>
    %4084 = stablehlo.broadcast_in_dim %4083, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4085 = stablehlo.multiply %4075, %4084 : tensor<1x8x1024xf32>
    %4086 = stablehlo.convert %4085 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4087 = stablehlo.convert %arg366 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4088 = stablehlo.broadcast_in_dim %4087, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4089 = stablehlo.broadcast_in_dim %4088, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4090 = stablehlo.multiply %4089, %4086 : tensor<1x8x1024xbf16>
    %4091 = stablehlo.convert %arg92 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4092 = stablehlo.convert %arg93 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %4093 = stablehlo.dot_general %4090, %4091, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4094 = stablehlo.dot_general %4093, %4092, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4095 = stablehlo.convert %arg375 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4096 = stablehlo.dot_general %4090, %4095, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4097 = stablehlo.add %4094, %4096 : tensor<1x8x2048xbf16>
    %4098 = stablehlo.convert %arg372 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4099 = stablehlo.dot_general %4090, %4098, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4100 = stablehlo.convert %arg94 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4101 = stablehlo.convert %arg95 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %4102 = stablehlo.dot_general %4090, %4100, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4103 = stablehlo.dot_general %4102, %4101, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4104 = stablehlo.convert %arg376 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4105 = stablehlo.dot_general %4090, %4104, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4106 = stablehlo.add %4103, %4105 : tensor<1x8x1024xbf16>
    %4107 = stablehlo.reshape %4097 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4108 = stablehlo.convert %4107 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4109 = stablehlo.multiply %4108, %4108 : tensor<1x8x16x128xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4110 = stablehlo.reduce(%4109 init: %cst_172) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4111 = stablehlo.broadcast_in_dim %4110, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4112 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4113 = stablehlo.divide %4111, %4112 : tensor<1x8x16x1xf32>
    %4114 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4115 = stablehlo.add %4113, %4114 : tensor<1x8x16x1xf32>
    %4116 = stablehlo.rsqrt %4115 : tensor<1x8x16x1xf32>
    %4117 = stablehlo.broadcast_in_dim %4116, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4118 = stablehlo.multiply %4108, %4117 : tensor<1x8x16x128xf32>
    %4119 = stablehlo.convert %4118 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4120 = stablehlo.convert %arg374 : (tensor<128xf32>) -> tensor<128xbf16>
    %4121 = stablehlo.broadcast_in_dim %4120, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4122 = stablehlo.broadcast_in_dim %4121, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4123 = stablehlo.multiply %4122, %4119 : tensor<1x8x16x128xbf16>
    %4124 = stablehlo.reshape %4099 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4125 = stablehlo.convert %4124 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4126 = stablehlo.multiply %4125, %4125 : tensor<1x8x8x128xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4127 = stablehlo.reduce(%4126 init: %cst_173) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4128 = stablehlo.broadcast_in_dim %4127, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4129 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4130 = stablehlo.divide %4128, %4129 : tensor<1x8x8x1xf32>
    %4131 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4132 = stablehlo.add %4130, %4131 : tensor<1x8x8x1xf32>
    %4133 = stablehlo.rsqrt %4132 : tensor<1x8x8x1xf32>
    %4134 = stablehlo.broadcast_in_dim %4133, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4135 = stablehlo.multiply %4125, %4134 : tensor<1x8x8x128xf32>
    %4136 = stablehlo.convert %4135 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4137 = stablehlo.convert %arg371 : (tensor<128xf32>) -> tensor<128xbf16>
    %4138 = stablehlo.broadcast_in_dim %4137, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4139 = stablehlo.broadcast_in_dim %4138, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4140 = stablehlo.multiply %4139, %4136 : tensor<1x8x8x128xbf16>
    %4141 = stablehlo.reshape %4106 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4142 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4143 = stablehlo.compare  LT, %7, %4142,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4144 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4145 = stablehlo.add %7, %4144 : tensor<1x8xi32>
    %4146 = stablehlo.select %4143, %4145, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4147 = stablehlo.broadcast_in_dim %4146, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4148 = "stablehlo.gather"(%26, %4147) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4149 = stablehlo.slice %4148 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4150 = stablehlo.slice %4148 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4151 = stablehlo.broadcast_in_dim %4149, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4152 = stablehlo.convert %4151 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4153 = stablehlo.broadcast_in_dim %4150, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4154 = stablehlo.convert %4153 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4155 = stablehlo.slice %4123 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4156 = stablehlo.slice %4123 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4157 = stablehlo.broadcast_in_dim %4152, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4158 = stablehlo.multiply %4155, %4157 : tensor<1x8x16x64xbf16>
    %4159 = stablehlo.broadcast_in_dim %4154, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4160 = stablehlo.multiply %4156, %4159 : tensor<1x8x16x64xbf16>
    %4161 = stablehlo.subtract %4158, %4160 : tensor<1x8x16x64xbf16>
    %4162 = stablehlo.broadcast_in_dim %4152, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4163 = stablehlo.multiply %4156, %4162 : tensor<1x8x16x64xbf16>
    %4164 = stablehlo.broadcast_in_dim %4154, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4165 = stablehlo.multiply %4155, %4164 : tensor<1x8x16x64xbf16>
    %4166 = stablehlo.add %4163, %4165 : tensor<1x8x16x64xbf16>
    %4167 = stablehlo.concatenate %4161, %4166, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4168 = stablehlo.broadcast_in_dim %4149, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4169 = stablehlo.convert %4168 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4170 = stablehlo.broadcast_in_dim %4150, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4171 = stablehlo.convert %4170 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4172 = stablehlo.slice %4140 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4173 = stablehlo.slice %4140 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4174 = stablehlo.broadcast_in_dim %4169, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4175 = stablehlo.multiply %4172, %4174 : tensor<1x8x8x64xbf16>
    %4176 = stablehlo.broadcast_in_dim %4171, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4177 = stablehlo.multiply %4173, %4176 : tensor<1x8x8x64xbf16>
    %4178 = stablehlo.subtract %4175, %4177 : tensor<1x8x8x64xbf16>
    %4179 = stablehlo.broadcast_in_dim %4169, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4180 = stablehlo.multiply %4173, %4179 : tensor<1x8x8x64xbf16>
    %4181 = stablehlo.broadcast_in_dim %4171, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4182 = stablehlo.multiply %4172, %4181 : tensor<1x8x8x64xbf16>
    %4183 = stablehlo.add %4180, %4182 : tensor<1x8x8x64xbf16>
    %4184 = stablehlo.concatenate %4178, %4183, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4185 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4186 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4187 = stablehlo.broadcast_in_dim %4185, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4188 = stablehlo.and %4187, %4186 : tensor<1x1x8x8xi1>
    %4189 = stablehlo.convert %4188 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4190 = sdy.sharding_constraint %4167 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4191 = sdy.sharding_constraint %4184 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4192 = sdy.sharding_constraint %4141 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4193 = sdy.sharding_constraint %4189 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4194 = stablehlo.reshape %4190 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4195 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4196 = stablehlo.multiply %4194, %4195 : tensor<1x8x8x2x128xbf16>
    %4197 = stablehlo.dot_general %4191, %4196, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4198 = stablehlo.transpose %4197, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4199 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4200 = stablehlo.compare  NE, %4193, %4199,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4201 = stablehlo.convert %4200 : tensor<1x1x8x8xi1>
    %4202 = stablehlo.broadcast_in_dim %4201, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4203 = stablehlo.reshape %4202 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4204 = call @_where_91(%4203, %4198, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4205 = stablehlo.convert %4204 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_175 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4206 = stablehlo.reduce(%4205 init: %cst_175) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4207 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4208 = stablehlo.maximum %4207, %4206 : tensor<1x8x2x8xf32>
    %4209 = stablehlo.broadcast_in_dim %4208, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4210 = stablehlo.broadcast_in_dim %4209, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4211 = stablehlo.subtract %4205, %4210 : tensor<1x8x2x8x8xf32>
    %4212 = stablehlo.exponential %4211 : tensor<1x8x2x8x8xf32>
    %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4213 = stablehlo.reduce(%4212 init: %cst_176) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4214 = stablehlo.broadcast_in_dim %4213, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4215 = stablehlo.broadcast_in_dim %4214, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4216 = stablehlo.divide %4212, %4215 : tensor<1x8x2x8x8xf32>
    %4217 = stablehlo.convert %4216 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4218 = stablehlo.dot_general %4192, %4217, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4219 = stablehlo.transpose %4218, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4220 = stablehlo.reshape %4219 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4221 = sdy.sharding_constraint %4220 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4222 = stablehlo.reshape %4221 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4223 = stablehlo.convert %arg373 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4224 = stablehlo.dot_general %4222, %4223, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4225 = stablehlo.add %4074, %4224 : tensor<1x8x1024xbf16>
    %4226 = stablehlo.convert %4225 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4227 = stablehlo.multiply %4226, %4226 : tensor<1x8x1024xf32>
    %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4228 = stablehlo.reduce(%4227 init: %cst_177) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4229 = stablehlo.broadcast_in_dim %4228, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4230 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4231 = stablehlo.divide %4229, %4230 : tensor<1x8x1xf32>
    %4232 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4233 = stablehlo.add %4231, %4232 : tensor<1x8x1xf32>
    %4234 = stablehlo.rsqrt %4233 : tensor<1x8x1xf32>
    %4235 = stablehlo.broadcast_in_dim %4234, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4236 = stablehlo.multiply %4226, %4235 : tensor<1x8x1024xf32>
    %4237 = stablehlo.convert %4236 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4238 = stablehlo.convert %arg370 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4239 = stablehlo.broadcast_in_dim %4238, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4240 = stablehlo.broadcast_in_dim %4239, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4241 = stablehlo.multiply %4240, %4237 : tensor<1x8x1024xbf16>
    %4242 = stablehlo.convert %arg368 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4243 = stablehlo.dot_general %4241, %4242, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4244 = call @silu(%4243) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4245 = stablehlo.convert %arg369 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4246 = stablehlo.dot_general %4241, %4245, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4247 = stablehlo.multiply %4244, %4246 : tensor<1x8x3072xbf16>
    %4248 = stablehlo.convert %arg367 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4249 = stablehlo.dot_general %4247, %4248, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4250 = stablehlo.add %4225, %4249 : tensor<1x8x1024xbf16>
    %4251 = stablehlo.convert %4250 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4252 = stablehlo.multiply %4251, %4251 : tensor<1x8x1024xf32>
    %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4253 = stablehlo.reduce(%4252 init: %cst_178) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4254 = stablehlo.broadcast_in_dim %4253, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4255 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4256 = stablehlo.divide %4254, %4255 : tensor<1x8x1xf32>
    %4257 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4258 = stablehlo.add %4256, %4257 : tensor<1x8x1xf32>
    %4259 = stablehlo.rsqrt %4258 : tensor<1x8x1xf32>
    %4260 = stablehlo.broadcast_in_dim %4259, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4261 = stablehlo.multiply %4251, %4260 : tensor<1x8x1024xf32>
    %4262 = stablehlo.convert %4261 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4263 = stablehlo.convert %arg377 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4264 = stablehlo.broadcast_in_dim %4263, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4265 = stablehlo.broadcast_in_dim %4264, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4266 = stablehlo.multiply %4265, %4262 : tensor<1x8x1024xbf16>
    %4267 = stablehlo.convert %arg96 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4268 = stablehlo.convert %arg97 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %4269 = stablehlo.dot_general %4266, %4267, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4270 = stablehlo.dot_general %4269, %4268, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4271 = stablehlo.convert %arg386 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4272 = stablehlo.dot_general %4266, %4271, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4273 = stablehlo.add %4270, %4272 : tensor<1x8x2048xbf16>
    %4274 = stablehlo.convert %arg383 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4275 = stablehlo.dot_general %4266, %4274, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4276 = stablehlo.convert %arg98 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4277 = stablehlo.convert %arg99 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %4278 = stablehlo.dot_general %4266, %4276, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4279 = stablehlo.dot_general %4278, %4277, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4280 = stablehlo.convert %arg387 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4281 = stablehlo.dot_general %4266, %4280, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4282 = stablehlo.add %4279, %4281 : tensor<1x8x1024xbf16>
    %4283 = stablehlo.reshape %4273 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4284 = stablehlo.convert %4283 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4285 = stablehlo.multiply %4284, %4284 : tensor<1x8x16x128xf32>
    %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4286 = stablehlo.reduce(%4285 init: %cst_179) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4287 = stablehlo.broadcast_in_dim %4286, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4288 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4289 = stablehlo.divide %4287, %4288 : tensor<1x8x16x1xf32>
    %4290 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4291 = stablehlo.add %4289, %4290 : tensor<1x8x16x1xf32>
    %4292 = stablehlo.rsqrt %4291 : tensor<1x8x16x1xf32>
    %4293 = stablehlo.broadcast_in_dim %4292, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4294 = stablehlo.multiply %4284, %4293 : tensor<1x8x16x128xf32>
    %4295 = stablehlo.convert %4294 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4296 = stablehlo.convert %arg385 : (tensor<128xf32>) -> tensor<128xbf16>
    %4297 = stablehlo.broadcast_in_dim %4296, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4298 = stablehlo.broadcast_in_dim %4297, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4299 = stablehlo.multiply %4298, %4295 : tensor<1x8x16x128xbf16>
    %4300 = stablehlo.reshape %4275 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4301 = stablehlo.convert %4300 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4302 = stablehlo.multiply %4301, %4301 : tensor<1x8x8x128xf32>
    %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4303 = stablehlo.reduce(%4302 init: %cst_180) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4304 = stablehlo.broadcast_in_dim %4303, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4305 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4306 = stablehlo.divide %4304, %4305 : tensor<1x8x8x1xf32>
    %4307 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4308 = stablehlo.add %4306, %4307 : tensor<1x8x8x1xf32>
    %4309 = stablehlo.rsqrt %4308 : tensor<1x8x8x1xf32>
    %4310 = stablehlo.broadcast_in_dim %4309, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4311 = stablehlo.multiply %4301, %4310 : tensor<1x8x8x128xf32>
    %4312 = stablehlo.convert %4311 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4313 = stablehlo.convert %arg382 : (tensor<128xf32>) -> tensor<128xbf16>
    %4314 = stablehlo.broadcast_in_dim %4313, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4315 = stablehlo.broadcast_in_dim %4314, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4316 = stablehlo.multiply %4315, %4312 : tensor<1x8x8x128xbf16>
    %4317 = stablehlo.reshape %4282 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4318 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4319 = stablehlo.compare  LT, %7, %4318,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4320 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4321 = stablehlo.add %7, %4320 : tensor<1x8xi32>
    %4322 = stablehlo.select %4319, %4321, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4323 = stablehlo.broadcast_in_dim %4322, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4324 = "stablehlo.gather"(%26, %4323) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4325 = stablehlo.slice %4324 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4326 = stablehlo.slice %4324 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4327 = stablehlo.broadcast_in_dim %4325, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4328 = stablehlo.convert %4327 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4329 = stablehlo.broadcast_in_dim %4326, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4330 = stablehlo.convert %4329 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4331 = stablehlo.slice %4299 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4332 = stablehlo.slice %4299 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4333 = stablehlo.broadcast_in_dim %4328, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4334 = stablehlo.multiply %4331, %4333 : tensor<1x8x16x64xbf16>
    %4335 = stablehlo.broadcast_in_dim %4330, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4336 = stablehlo.multiply %4332, %4335 : tensor<1x8x16x64xbf16>
    %4337 = stablehlo.subtract %4334, %4336 : tensor<1x8x16x64xbf16>
    %4338 = stablehlo.broadcast_in_dim %4328, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4339 = stablehlo.multiply %4332, %4338 : tensor<1x8x16x64xbf16>
    %4340 = stablehlo.broadcast_in_dim %4330, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4341 = stablehlo.multiply %4331, %4340 : tensor<1x8x16x64xbf16>
    %4342 = stablehlo.add %4339, %4341 : tensor<1x8x16x64xbf16>
    %4343 = stablehlo.concatenate %4337, %4342, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4344 = stablehlo.broadcast_in_dim %4325, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4345 = stablehlo.convert %4344 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4346 = stablehlo.broadcast_in_dim %4326, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4347 = stablehlo.convert %4346 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4348 = stablehlo.slice %4316 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4349 = stablehlo.slice %4316 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4350 = stablehlo.broadcast_in_dim %4345, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4351 = stablehlo.multiply %4348, %4350 : tensor<1x8x8x64xbf16>
    %4352 = stablehlo.broadcast_in_dim %4347, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4353 = stablehlo.multiply %4349, %4352 : tensor<1x8x8x64xbf16>
    %4354 = stablehlo.subtract %4351, %4353 : tensor<1x8x8x64xbf16>
    %4355 = stablehlo.broadcast_in_dim %4345, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4356 = stablehlo.multiply %4349, %4355 : tensor<1x8x8x64xbf16>
    %4357 = stablehlo.broadcast_in_dim %4347, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4358 = stablehlo.multiply %4348, %4357 : tensor<1x8x8x64xbf16>
    %4359 = stablehlo.add %4356, %4358 : tensor<1x8x8x64xbf16>
    %4360 = stablehlo.concatenate %4354, %4359, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4361 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4362 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4363 = stablehlo.broadcast_in_dim %4361, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4364 = stablehlo.and %4363, %4362 : tensor<1x1x8x8xi1>
    %4365 = stablehlo.convert %4364 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4366 = sdy.sharding_constraint %4343 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4367 = sdy.sharding_constraint %4360 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4368 = sdy.sharding_constraint %4317 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4369 = sdy.sharding_constraint %4365 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4370 = stablehlo.reshape %4366 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4371 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4372 = stablehlo.multiply %4370, %4371 : tensor<1x8x8x2x128xbf16>
    %4373 = stablehlo.dot_general %4367, %4372, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4374 = stablehlo.transpose %4373, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4375 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4376 = stablehlo.compare  NE, %4369, %4375,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4377 = stablehlo.convert %4376 : tensor<1x1x8x8xi1>
    %4378 = stablehlo.broadcast_in_dim %4377, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4379 = stablehlo.reshape %4378 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4380 = call @_where_91(%4379, %4374, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4381 = stablehlo.convert %4380 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_182 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4382 = stablehlo.reduce(%4381 init: %cst_182) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4383 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4384 = stablehlo.maximum %4383, %4382 : tensor<1x8x2x8xf32>
    %4385 = stablehlo.broadcast_in_dim %4384, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4386 = stablehlo.broadcast_in_dim %4385, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4387 = stablehlo.subtract %4381, %4386 : tensor<1x8x2x8x8xf32>
    %4388 = stablehlo.exponential %4387 : tensor<1x8x2x8x8xf32>
    %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4389 = stablehlo.reduce(%4388 init: %cst_183) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4390 = stablehlo.broadcast_in_dim %4389, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4391 = stablehlo.broadcast_in_dim %4390, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4392 = stablehlo.divide %4388, %4391 : tensor<1x8x2x8x8xf32>
    %4393 = stablehlo.convert %4392 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4394 = stablehlo.dot_general %4368, %4393, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4395 = stablehlo.transpose %4394, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4396 = stablehlo.reshape %4395 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4397 = sdy.sharding_constraint %4396 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4398 = stablehlo.reshape %4397 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4399 = stablehlo.convert %arg384 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4400 = stablehlo.dot_general %4398, %4399, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4401 = stablehlo.add %4250, %4400 : tensor<1x8x1024xbf16>
    %4402 = stablehlo.convert %4401 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4403 = stablehlo.multiply %4402, %4402 : tensor<1x8x1024xf32>
    %cst_184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4404 = stablehlo.reduce(%4403 init: %cst_184) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4405 = stablehlo.broadcast_in_dim %4404, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4406 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4407 = stablehlo.divide %4405, %4406 : tensor<1x8x1xf32>
    %4408 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4409 = stablehlo.add %4407, %4408 : tensor<1x8x1xf32>
    %4410 = stablehlo.rsqrt %4409 : tensor<1x8x1xf32>
    %4411 = stablehlo.broadcast_in_dim %4410, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4412 = stablehlo.multiply %4402, %4411 : tensor<1x8x1024xf32>
    %4413 = stablehlo.convert %4412 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4414 = stablehlo.convert %arg381 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4415 = stablehlo.broadcast_in_dim %4414, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4416 = stablehlo.broadcast_in_dim %4415, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4417 = stablehlo.multiply %4416, %4413 : tensor<1x8x1024xbf16>
    %4418 = stablehlo.convert %arg379 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4419 = stablehlo.dot_general %4417, %4418, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4420 = call @silu(%4419) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4421 = stablehlo.convert %arg380 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4422 = stablehlo.dot_general %4417, %4421, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4423 = stablehlo.multiply %4420, %4422 : tensor<1x8x3072xbf16>
    %4424 = stablehlo.convert %arg378 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4425 = stablehlo.dot_general %4423, %4424, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4426 = stablehlo.add %4401, %4425 : tensor<1x8x1024xbf16>
    %4427 = stablehlo.convert %4426 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4428 = stablehlo.multiply %4427, %4427 : tensor<1x8x1024xf32>
    %cst_185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4429 = stablehlo.reduce(%4428 init: %cst_185) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4430 = stablehlo.broadcast_in_dim %4429, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4431 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4432 = stablehlo.divide %4430, %4431 : tensor<1x8x1xf32>
    %4433 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4434 = stablehlo.add %4432, %4433 : tensor<1x8x1xf32>
    %4435 = stablehlo.rsqrt %4434 : tensor<1x8x1xf32>
    %4436 = stablehlo.broadcast_in_dim %4435, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4437 = stablehlo.multiply %4427, %4436 : tensor<1x8x1024xf32>
    %4438 = stablehlo.convert %4437 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4439 = stablehlo.convert %arg388 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4440 = stablehlo.broadcast_in_dim %4439, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4441 = stablehlo.broadcast_in_dim %4440, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4442 = stablehlo.multiply %4441, %4438 : tensor<1x8x1024xbf16>
    %4443 = stablehlo.convert %arg100 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4444 = stablehlo.convert %arg101 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %4445 = stablehlo.dot_general %4442, %4443, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4446 = stablehlo.dot_general %4445, %4444, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4447 = stablehlo.convert %arg397 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4448 = stablehlo.dot_general %4442, %4447, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4449 = stablehlo.add %4446, %4448 : tensor<1x8x2048xbf16>
    %4450 = stablehlo.convert %arg394 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4451 = stablehlo.dot_general %4442, %4450, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4452 = stablehlo.convert %arg102 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4453 = stablehlo.convert %arg103 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %4454 = stablehlo.dot_general %4442, %4452, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4455 = stablehlo.dot_general %4454, %4453, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4456 = stablehlo.convert %arg398 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4457 = stablehlo.dot_general %4442, %4456, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4458 = stablehlo.add %4455, %4457 : tensor<1x8x1024xbf16>
    %4459 = stablehlo.reshape %4449 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4460 = stablehlo.convert %4459 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4461 = stablehlo.multiply %4460, %4460 : tensor<1x8x16x128xf32>
    %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4462 = stablehlo.reduce(%4461 init: %cst_186) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4463 = stablehlo.broadcast_in_dim %4462, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4464 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4465 = stablehlo.divide %4463, %4464 : tensor<1x8x16x1xf32>
    %4466 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4467 = stablehlo.add %4465, %4466 : tensor<1x8x16x1xf32>
    %4468 = stablehlo.rsqrt %4467 : tensor<1x8x16x1xf32>
    %4469 = stablehlo.broadcast_in_dim %4468, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4470 = stablehlo.multiply %4460, %4469 : tensor<1x8x16x128xf32>
    %4471 = stablehlo.convert %4470 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4472 = stablehlo.convert %arg396 : (tensor<128xf32>) -> tensor<128xbf16>
    %4473 = stablehlo.broadcast_in_dim %4472, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4474 = stablehlo.broadcast_in_dim %4473, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4475 = stablehlo.multiply %4474, %4471 : tensor<1x8x16x128xbf16>
    %4476 = stablehlo.reshape %4451 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4477 = stablehlo.convert %4476 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4478 = stablehlo.multiply %4477, %4477 : tensor<1x8x8x128xf32>
    %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4479 = stablehlo.reduce(%4478 init: %cst_187) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4480 = stablehlo.broadcast_in_dim %4479, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4481 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4482 = stablehlo.divide %4480, %4481 : tensor<1x8x8x1xf32>
    %4483 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4484 = stablehlo.add %4482, %4483 : tensor<1x8x8x1xf32>
    %4485 = stablehlo.rsqrt %4484 : tensor<1x8x8x1xf32>
    %4486 = stablehlo.broadcast_in_dim %4485, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4487 = stablehlo.multiply %4477, %4486 : tensor<1x8x8x128xf32>
    %4488 = stablehlo.convert %4487 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4489 = stablehlo.convert %arg393 : (tensor<128xf32>) -> tensor<128xbf16>
    %4490 = stablehlo.broadcast_in_dim %4489, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4491 = stablehlo.broadcast_in_dim %4490, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4492 = stablehlo.multiply %4491, %4488 : tensor<1x8x8x128xbf16>
    %4493 = stablehlo.reshape %4458 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4494 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4495 = stablehlo.compare  LT, %7, %4494,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4496 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4497 = stablehlo.add %7, %4496 : tensor<1x8xi32>
    %4498 = stablehlo.select %4495, %4497, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4499 = stablehlo.broadcast_in_dim %4498, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4500 = "stablehlo.gather"(%26, %4499) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4501 = stablehlo.slice %4500 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4502 = stablehlo.slice %4500 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4503 = stablehlo.broadcast_in_dim %4501, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4504 = stablehlo.convert %4503 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4505 = stablehlo.broadcast_in_dim %4502, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4506 = stablehlo.convert %4505 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4507 = stablehlo.slice %4475 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4508 = stablehlo.slice %4475 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4509 = stablehlo.broadcast_in_dim %4504, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4510 = stablehlo.multiply %4507, %4509 : tensor<1x8x16x64xbf16>
    %4511 = stablehlo.broadcast_in_dim %4506, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4512 = stablehlo.multiply %4508, %4511 : tensor<1x8x16x64xbf16>
    %4513 = stablehlo.subtract %4510, %4512 : tensor<1x8x16x64xbf16>
    %4514 = stablehlo.broadcast_in_dim %4504, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4515 = stablehlo.multiply %4508, %4514 : tensor<1x8x16x64xbf16>
    %4516 = stablehlo.broadcast_in_dim %4506, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4517 = stablehlo.multiply %4507, %4516 : tensor<1x8x16x64xbf16>
    %4518 = stablehlo.add %4515, %4517 : tensor<1x8x16x64xbf16>
    %4519 = stablehlo.concatenate %4513, %4518, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4520 = stablehlo.broadcast_in_dim %4501, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4521 = stablehlo.convert %4520 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4522 = stablehlo.broadcast_in_dim %4502, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4523 = stablehlo.convert %4522 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4524 = stablehlo.slice %4492 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4525 = stablehlo.slice %4492 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4526 = stablehlo.broadcast_in_dim %4521, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4527 = stablehlo.multiply %4524, %4526 : tensor<1x8x8x64xbf16>
    %4528 = stablehlo.broadcast_in_dim %4523, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4529 = stablehlo.multiply %4525, %4528 : tensor<1x8x8x64xbf16>
    %4530 = stablehlo.subtract %4527, %4529 : tensor<1x8x8x64xbf16>
    %4531 = stablehlo.broadcast_in_dim %4521, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4532 = stablehlo.multiply %4525, %4531 : tensor<1x8x8x64xbf16>
    %4533 = stablehlo.broadcast_in_dim %4523, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4534 = stablehlo.multiply %4524, %4533 : tensor<1x8x8x64xbf16>
    %4535 = stablehlo.add %4532, %4534 : tensor<1x8x8x64xbf16>
    %4536 = stablehlo.concatenate %4530, %4535, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4537 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4538 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4539 = stablehlo.broadcast_in_dim %4537, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4540 = stablehlo.and %4539, %4538 : tensor<1x1x8x8xi1>
    %4541 = stablehlo.convert %4540 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4542 = sdy.sharding_constraint %4519 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4543 = sdy.sharding_constraint %4536 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4544 = sdy.sharding_constraint %4493 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4545 = sdy.sharding_constraint %4541 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4546 = stablehlo.reshape %4542 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4547 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4548 = stablehlo.multiply %4546, %4547 : tensor<1x8x8x2x128xbf16>
    %4549 = stablehlo.dot_general %4543, %4548, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4550 = stablehlo.transpose %4549, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4551 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4552 = stablehlo.compare  NE, %4545, %4551,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4553 = stablehlo.convert %4552 : tensor<1x1x8x8xi1>
    %4554 = stablehlo.broadcast_in_dim %4553, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4555 = stablehlo.reshape %4554 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4556 = call @_where_91(%4555, %4550, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4557 = stablehlo.convert %4556 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_189 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4558 = stablehlo.reduce(%4557 init: %cst_189) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4559 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4560 = stablehlo.maximum %4559, %4558 : tensor<1x8x2x8xf32>
    %4561 = stablehlo.broadcast_in_dim %4560, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4562 = stablehlo.broadcast_in_dim %4561, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4563 = stablehlo.subtract %4557, %4562 : tensor<1x8x2x8x8xf32>
    %4564 = stablehlo.exponential %4563 : tensor<1x8x2x8x8xf32>
    %cst_190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4565 = stablehlo.reduce(%4564 init: %cst_190) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4566 = stablehlo.broadcast_in_dim %4565, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4567 = stablehlo.broadcast_in_dim %4566, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4568 = stablehlo.divide %4564, %4567 : tensor<1x8x2x8x8xf32>
    %4569 = stablehlo.convert %4568 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4570 = stablehlo.dot_general %4544, %4569, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4571 = stablehlo.transpose %4570, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4572 = stablehlo.reshape %4571 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4573 = sdy.sharding_constraint %4572 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4574 = stablehlo.reshape %4573 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4575 = stablehlo.convert %arg395 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4576 = stablehlo.dot_general %4574, %4575, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4577 = stablehlo.add %4426, %4576 : tensor<1x8x1024xbf16>
    %4578 = stablehlo.convert %4577 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4579 = stablehlo.multiply %4578, %4578 : tensor<1x8x1024xf32>
    %cst_191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4580 = stablehlo.reduce(%4579 init: %cst_191) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4581 = stablehlo.broadcast_in_dim %4580, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4582 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4583 = stablehlo.divide %4581, %4582 : tensor<1x8x1xf32>
    %4584 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4585 = stablehlo.add %4583, %4584 : tensor<1x8x1xf32>
    %4586 = stablehlo.rsqrt %4585 : tensor<1x8x1xf32>
    %4587 = stablehlo.broadcast_in_dim %4586, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4588 = stablehlo.multiply %4578, %4587 : tensor<1x8x1024xf32>
    %4589 = stablehlo.convert %4588 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4590 = stablehlo.convert %arg392 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4591 = stablehlo.broadcast_in_dim %4590, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4592 = stablehlo.broadcast_in_dim %4591, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4593 = stablehlo.multiply %4592, %4589 : tensor<1x8x1024xbf16>
    %4594 = stablehlo.convert %arg390 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4595 = stablehlo.dot_general %4593, %4594, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4596 = call @silu(%4595) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4597 = stablehlo.convert %arg391 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4598 = stablehlo.dot_general %4593, %4597, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4599 = stablehlo.multiply %4596, %4598 : tensor<1x8x3072xbf16>
    %4600 = stablehlo.convert %arg389 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4601 = stablehlo.dot_general %4599, %4600, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4602 = stablehlo.add %4577, %4601 : tensor<1x8x1024xbf16>
    %4603 = stablehlo.convert %4602 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4604 = stablehlo.multiply %4603, %4603 : tensor<1x8x1024xf32>
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4605 = stablehlo.reduce(%4604 init: %cst_192) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4606 = stablehlo.broadcast_in_dim %4605, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4607 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4608 = stablehlo.divide %4606, %4607 : tensor<1x8x1xf32>
    %4609 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4610 = stablehlo.add %4608, %4609 : tensor<1x8x1xf32>
    %4611 = stablehlo.rsqrt %4610 : tensor<1x8x1xf32>
    %4612 = stablehlo.broadcast_in_dim %4611, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4613 = stablehlo.multiply %4603, %4612 : tensor<1x8x1024xf32>
    %4614 = stablehlo.convert %4613 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4615 = stablehlo.convert %arg399 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4616 = stablehlo.broadcast_in_dim %4615, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4617 = stablehlo.broadcast_in_dim %4616, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4618 = stablehlo.multiply %4617, %4614 : tensor<1x8x1024xbf16>
    %4619 = stablehlo.convert %arg104 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4620 = stablehlo.convert %arg105 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %4621 = stablehlo.dot_general %4618, %4619, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4622 = stablehlo.dot_general %4621, %4620, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4623 = stablehlo.convert %arg408 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4624 = stablehlo.dot_general %4618, %4623, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4625 = stablehlo.add %4622, %4624 : tensor<1x8x2048xbf16>
    %4626 = stablehlo.convert %arg405 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4627 = stablehlo.dot_general %4618, %4626, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4628 = stablehlo.convert %arg106 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4629 = stablehlo.convert %arg107 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %4630 = stablehlo.dot_general %4618, %4628, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4631 = stablehlo.dot_general %4630, %4629, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4632 = stablehlo.convert %arg409 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4633 = stablehlo.dot_general %4618, %4632, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4634 = stablehlo.add %4631, %4633 : tensor<1x8x1024xbf16>
    %4635 = stablehlo.reshape %4625 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4636 = stablehlo.convert %4635 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4637 = stablehlo.multiply %4636, %4636 : tensor<1x8x16x128xf32>
    %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4638 = stablehlo.reduce(%4637 init: %cst_193) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4639 = stablehlo.broadcast_in_dim %4638, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4640 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4641 = stablehlo.divide %4639, %4640 : tensor<1x8x16x1xf32>
    %4642 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4643 = stablehlo.add %4641, %4642 : tensor<1x8x16x1xf32>
    %4644 = stablehlo.rsqrt %4643 : tensor<1x8x16x1xf32>
    %4645 = stablehlo.broadcast_in_dim %4644, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4646 = stablehlo.multiply %4636, %4645 : tensor<1x8x16x128xf32>
    %4647 = stablehlo.convert %4646 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4648 = stablehlo.convert %arg407 : (tensor<128xf32>) -> tensor<128xbf16>
    %4649 = stablehlo.broadcast_in_dim %4648, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4650 = stablehlo.broadcast_in_dim %4649, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4651 = stablehlo.multiply %4650, %4647 : tensor<1x8x16x128xbf16>
    %4652 = stablehlo.reshape %4627 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4653 = stablehlo.convert %4652 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4654 = stablehlo.multiply %4653, %4653 : tensor<1x8x8x128xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4655 = stablehlo.reduce(%4654 init: %cst_194) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4656 = stablehlo.broadcast_in_dim %4655, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4657 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4658 = stablehlo.divide %4656, %4657 : tensor<1x8x8x1xf32>
    %4659 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4660 = stablehlo.add %4658, %4659 : tensor<1x8x8x1xf32>
    %4661 = stablehlo.rsqrt %4660 : tensor<1x8x8x1xf32>
    %4662 = stablehlo.broadcast_in_dim %4661, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4663 = stablehlo.multiply %4653, %4662 : tensor<1x8x8x128xf32>
    %4664 = stablehlo.convert %4663 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4665 = stablehlo.convert %arg404 : (tensor<128xf32>) -> tensor<128xbf16>
    %4666 = stablehlo.broadcast_in_dim %4665, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4667 = stablehlo.broadcast_in_dim %4666, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4668 = stablehlo.multiply %4667, %4664 : tensor<1x8x8x128xbf16>
    %4669 = stablehlo.reshape %4634 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4670 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4671 = stablehlo.compare  LT, %7, %4670,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4672 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4673 = stablehlo.add %7, %4672 : tensor<1x8xi32>
    %4674 = stablehlo.select %4671, %4673, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4675 = stablehlo.broadcast_in_dim %4674, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4676 = "stablehlo.gather"(%26, %4675) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4677 = stablehlo.slice %4676 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4678 = stablehlo.slice %4676 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4679 = stablehlo.broadcast_in_dim %4677, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4680 = stablehlo.convert %4679 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4681 = stablehlo.broadcast_in_dim %4678, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4682 = stablehlo.convert %4681 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4683 = stablehlo.slice %4651 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4684 = stablehlo.slice %4651 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4685 = stablehlo.broadcast_in_dim %4680, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4686 = stablehlo.multiply %4683, %4685 : tensor<1x8x16x64xbf16>
    %4687 = stablehlo.broadcast_in_dim %4682, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4688 = stablehlo.multiply %4684, %4687 : tensor<1x8x16x64xbf16>
    %4689 = stablehlo.subtract %4686, %4688 : tensor<1x8x16x64xbf16>
    %4690 = stablehlo.broadcast_in_dim %4680, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4691 = stablehlo.multiply %4684, %4690 : tensor<1x8x16x64xbf16>
    %4692 = stablehlo.broadcast_in_dim %4682, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4693 = stablehlo.multiply %4683, %4692 : tensor<1x8x16x64xbf16>
    %4694 = stablehlo.add %4691, %4693 : tensor<1x8x16x64xbf16>
    %4695 = stablehlo.concatenate %4689, %4694, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4696 = stablehlo.broadcast_in_dim %4677, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4697 = stablehlo.convert %4696 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4698 = stablehlo.broadcast_in_dim %4678, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4699 = stablehlo.convert %4698 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4700 = stablehlo.slice %4668 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4701 = stablehlo.slice %4668 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4702 = stablehlo.broadcast_in_dim %4697, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4703 = stablehlo.multiply %4700, %4702 : tensor<1x8x8x64xbf16>
    %4704 = stablehlo.broadcast_in_dim %4699, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4705 = stablehlo.multiply %4701, %4704 : tensor<1x8x8x64xbf16>
    %4706 = stablehlo.subtract %4703, %4705 : tensor<1x8x8x64xbf16>
    %4707 = stablehlo.broadcast_in_dim %4697, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4708 = stablehlo.multiply %4701, %4707 : tensor<1x8x8x64xbf16>
    %4709 = stablehlo.broadcast_in_dim %4699, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4710 = stablehlo.multiply %4700, %4709 : tensor<1x8x8x64xbf16>
    %4711 = stablehlo.add %4708, %4710 : tensor<1x8x8x64xbf16>
    %4712 = stablehlo.concatenate %4706, %4711, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4713 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4714 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4715 = stablehlo.broadcast_in_dim %4713, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4716 = stablehlo.and %4715, %4714 : tensor<1x1x8x8xi1>
    %4717 = stablehlo.convert %4716 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4718 = sdy.sharding_constraint %4695 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4719 = sdy.sharding_constraint %4712 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4720 = sdy.sharding_constraint %4669 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4721 = sdy.sharding_constraint %4717 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4722 = stablehlo.reshape %4718 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4723 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4724 = stablehlo.multiply %4722, %4723 : tensor<1x8x8x2x128xbf16>
    %4725 = stablehlo.dot_general %4719, %4724, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4726 = stablehlo.transpose %4725, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4727 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4728 = stablehlo.compare  NE, %4721, %4727,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4729 = stablehlo.convert %4728 : tensor<1x1x8x8xi1>
    %4730 = stablehlo.broadcast_in_dim %4729, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4731 = stablehlo.reshape %4730 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4732 = call @_where_91(%4731, %4726, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4733 = stablehlo.convert %4732 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_196 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4734 = stablehlo.reduce(%4733 init: %cst_196) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4735 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4736 = stablehlo.maximum %4735, %4734 : tensor<1x8x2x8xf32>
    %4737 = stablehlo.broadcast_in_dim %4736, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4738 = stablehlo.broadcast_in_dim %4737, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4739 = stablehlo.subtract %4733, %4738 : tensor<1x8x2x8x8xf32>
    %4740 = stablehlo.exponential %4739 : tensor<1x8x2x8x8xf32>
    %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4741 = stablehlo.reduce(%4740 init: %cst_197) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4742 = stablehlo.broadcast_in_dim %4741, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4743 = stablehlo.broadcast_in_dim %4742, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4744 = stablehlo.divide %4740, %4743 : tensor<1x8x2x8x8xf32>
    %4745 = stablehlo.convert %4744 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4746 = stablehlo.dot_general %4720, %4745, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4747 = stablehlo.transpose %4746, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4748 = stablehlo.reshape %4747 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4749 = sdy.sharding_constraint %4748 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4750 = stablehlo.reshape %4749 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4751 = stablehlo.convert %arg406 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4752 = stablehlo.dot_general %4750, %4751, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4753 = stablehlo.add %4602, %4752 : tensor<1x8x1024xbf16>
    %4754 = stablehlo.convert %4753 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4755 = stablehlo.multiply %4754, %4754 : tensor<1x8x1024xf32>
    %cst_198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4756 = stablehlo.reduce(%4755 init: %cst_198) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4757 = stablehlo.broadcast_in_dim %4756, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4758 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4759 = stablehlo.divide %4757, %4758 : tensor<1x8x1xf32>
    %4760 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4761 = stablehlo.add %4759, %4760 : tensor<1x8x1xf32>
    %4762 = stablehlo.rsqrt %4761 : tensor<1x8x1xf32>
    %4763 = stablehlo.broadcast_in_dim %4762, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4764 = stablehlo.multiply %4754, %4763 : tensor<1x8x1024xf32>
    %4765 = stablehlo.convert %4764 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4766 = stablehlo.convert %arg403 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4767 = stablehlo.broadcast_in_dim %4766, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4768 = stablehlo.broadcast_in_dim %4767, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4769 = stablehlo.multiply %4768, %4765 : tensor<1x8x1024xbf16>
    %4770 = stablehlo.convert %arg401 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4771 = stablehlo.dot_general %4769, %4770, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4772 = call @silu(%4771) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4773 = stablehlo.convert %arg402 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4774 = stablehlo.dot_general %4769, %4773, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4775 = stablehlo.multiply %4772, %4774 : tensor<1x8x3072xbf16>
    %4776 = stablehlo.convert %arg400 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4777 = stablehlo.dot_general %4775, %4776, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4778 = stablehlo.add %4753, %4777 : tensor<1x8x1024xbf16>
    %4779 = stablehlo.convert %4778 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4780 = stablehlo.multiply %4779, %4779 : tensor<1x8x1024xf32>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4781 = stablehlo.reduce(%4780 init: %cst_199) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4782 = stablehlo.broadcast_in_dim %4781, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4783 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4784 = stablehlo.divide %4782, %4783 : tensor<1x8x1xf32>
    %4785 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4786 = stablehlo.add %4784, %4785 : tensor<1x8x1xf32>
    %4787 = stablehlo.rsqrt %4786 : tensor<1x8x1xf32>
    %4788 = stablehlo.broadcast_in_dim %4787, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4789 = stablehlo.multiply %4779, %4788 : tensor<1x8x1024xf32>
    %4790 = stablehlo.convert %4789 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4791 = stablehlo.convert %arg410 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4792 = stablehlo.broadcast_in_dim %4791, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4793 = stablehlo.broadcast_in_dim %4792, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4794 = stablehlo.multiply %4793, %4790 : tensor<1x8x1024xbf16>
    %4795 = stablehlo.convert %arg108 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4796 = stablehlo.convert %arg109 : (tensor<16x2048xf32>) -> tensor<16x2048xbf16>
    %4797 = stablehlo.dot_general %4794, %4795, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4798 = stablehlo.dot_general %4797, %4796, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4799 = stablehlo.convert %arg419 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4800 = stablehlo.dot_general %4794, %4799, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4801 = stablehlo.add %4798, %4800 : tensor<1x8x2048xbf16>
    %4802 = stablehlo.convert %arg416 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4803 = stablehlo.dot_general %4794, %4802, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4804 = stablehlo.convert %arg110 : (tensor<1024x16xf32>) -> tensor<1024x16xbf16>
    %4805 = stablehlo.convert %arg111 : (tensor<16x1024xf32>) -> tensor<16x1024xbf16>
    %4806 = stablehlo.dot_general %4794, %4804, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x16xbf16>) -> tensor<1x8x16xbf16>
    %4807 = stablehlo.dot_general %4806, %4805, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x16xbf16>, tensor<16x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4808 = stablehlo.convert %arg420 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4809 = stablehlo.dot_general %4794, %4808, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4810 = stablehlo.add %4807, %4809 : tensor<1x8x1024xbf16>
    %4811 = stablehlo.reshape %4801 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4812 = stablehlo.convert %4811 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4813 = stablehlo.multiply %4812, %4812 : tensor<1x8x16x128xf32>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4814 = stablehlo.reduce(%4813 init: %cst_200) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4815 = stablehlo.broadcast_in_dim %4814, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4816 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4817 = stablehlo.divide %4815, %4816 : tensor<1x8x16x1xf32>
    %4818 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4819 = stablehlo.add %4817, %4818 : tensor<1x8x16x1xf32>
    %4820 = stablehlo.rsqrt %4819 : tensor<1x8x16x1xf32>
    %4821 = stablehlo.broadcast_in_dim %4820, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4822 = stablehlo.multiply %4812, %4821 : tensor<1x8x16x128xf32>
    %4823 = stablehlo.convert %4822 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4824 = stablehlo.convert %arg418 : (tensor<128xf32>) -> tensor<128xbf16>
    %4825 = stablehlo.broadcast_in_dim %4824, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4826 = stablehlo.broadcast_in_dim %4825, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4827 = stablehlo.multiply %4826, %4823 : tensor<1x8x16x128xbf16>
    %4828 = stablehlo.reshape %4803 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4829 = stablehlo.convert %4828 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4830 = stablehlo.multiply %4829, %4829 : tensor<1x8x8x128xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4831 = stablehlo.reduce(%4830 init: %cst_201) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4832 = stablehlo.broadcast_in_dim %4831, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4833 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4834 = stablehlo.divide %4832, %4833 : tensor<1x8x8x1xf32>
    %4835 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4836 = stablehlo.add %4834, %4835 : tensor<1x8x8x1xf32>
    %4837 = stablehlo.rsqrt %4836 : tensor<1x8x8x1xf32>
    %4838 = stablehlo.broadcast_in_dim %4837, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4839 = stablehlo.multiply %4829, %4838 : tensor<1x8x8x128xf32>
    %4840 = stablehlo.convert %4839 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4841 = stablehlo.convert %arg415 : (tensor<128xf32>) -> tensor<128xbf16>
    %4842 = stablehlo.broadcast_in_dim %4841, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4843 = stablehlo.broadcast_in_dim %4842, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4844 = stablehlo.multiply %4843, %4840 : tensor<1x8x8x128xbf16>
    %4845 = stablehlo.reshape %4810 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4846 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4847 = stablehlo.compare  LT, %7, %4846,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4848 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4849 = stablehlo.add %7, %4848 : tensor<1x8xi32>
    %4850 = stablehlo.select %4847, %4849, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4851 = stablehlo.broadcast_in_dim %4850, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4852 = "stablehlo.gather"(%26, %4851) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4853 = stablehlo.slice %4852 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4854 = stablehlo.slice %4852 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4855 = stablehlo.broadcast_in_dim %4853, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4856 = stablehlo.convert %4855 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4857 = stablehlo.broadcast_in_dim %4854, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4858 = stablehlo.convert %4857 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4859 = stablehlo.slice %4827 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4860 = stablehlo.slice %4827 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4861 = stablehlo.broadcast_in_dim %4856, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4862 = stablehlo.multiply %4859, %4861 : tensor<1x8x16x64xbf16>
    %4863 = stablehlo.broadcast_in_dim %4858, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4864 = stablehlo.multiply %4860, %4863 : tensor<1x8x16x64xbf16>
    %4865 = stablehlo.subtract %4862, %4864 : tensor<1x8x16x64xbf16>
    %4866 = stablehlo.broadcast_in_dim %4856, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4867 = stablehlo.multiply %4860, %4866 : tensor<1x8x16x64xbf16>
    %4868 = stablehlo.broadcast_in_dim %4858, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4869 = stablehlo.multiply %4859, %4868 : tensor<1x8x16x64xbf16>
    %4870 = stablehlo.add %4867, %4869 : tensor<1x8x16x64xbf16>
    %4871 = stablehlo.concatenate %4865, %4870, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4872 = stablehlo.broadcast_in_dim %4853, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4873 = stablehlo.convert %4872 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4874 = stablehlo.broadcast_in_dim %4854, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4875 = stablehlo.convert %4874 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4876 = stablehlo.slice %4844 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4877 = stablehlo.slice %4844 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4878 = stablehlo.broadcast_in_dim %4873, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4879 = stablehlo.multiply %4876, %4878 : tensor<1x8x8x64xbf16>
    %4880 = stablehlo.broadcast_in_dim %4875, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4881 = stablehlo.multiply %4877, %4880 : tensor<1x8x8x64xbf16>
    %4882 = stablehlo.subtract %4879, %4881 : tensor<1x8x8x64xbf16>
    %4883 = stablehlo.broadcast_in_dim %4873, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4884 = stablehlo.multiply %4877, %4883 : tensor<1x8x8x64xbf16>
    %4885 = stablehlo.broadcast_in_dim %4875, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4886 = stablehlo.multiply %4876, %4885 : tensor<1x8x8x64xbf16>
    %4887 = stablehlo.add %4884, %4886 : tensor<1x8x8x64xbf16>
    %4888 = stablehlo.concatenate %4882, %4887, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4889 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4890 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4891 = stablehlo.broadcast_in_dim %4889, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4892 = stablehlo.and %4891, %4890 : tensor<1x1x8x8xi1>
    %4893 = stablehlo.convert %4892 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4894 = sdy.sharding_constraint %4871 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4895 = sdy.sharding_constraint %4888 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4896 = sdy.sharding_constraint %4845 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4897 = sdy.sharding_constraint %4893 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4898 = stablehlo.reshape %4894 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4899 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4900 = stablehlo.multiply %4898, %4899 : tensor<1x8x8x2x128xbf16>
    %4901 = stablehlo.dot_general %4895, %4900, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4902 = stablehlo.transpose %4901, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4903 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4904 = stablehlo.compare  NE, %4897, %4903,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4905 = stablehlo.convert %4904 : tensor<1x1x8x8xi1>
    %4906 = stablehlo.broadcast_in_dim %4905, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4907 = stablehlo.reshape %4906 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4908 = call @_where_91(%4907, %4902, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4909 = stablehlo.convert %4908 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_203 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4910 = stablehlo.reduce(%4909 init: %cst_203) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4911 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4912 = stablehlo.maximum %4911, %4910 : tensor<1x8x2x8xf32>
    %4913 = stablehlo.broadcast_in_dim %4912, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4914 = stablehlo.broadcast_in_dim %4913, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4915 = stablehlo.subtract %4909, %4914 : tensor<1x8x2x8x8xf32>
    %4916 = stablehlo.exponential %4915 : tensor<1x8x2x8x8xf32>
    %cst_204 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4917 = stablehlo.reduce(%4916 init: %cst_204) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4918 = stablehlo.broadcast_in_dim %4917, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4919 = stablehlo.broadcast_in_dim %4918, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4920 = stablehlo.divide %4916, %4919 : tensor<1x8x2x8x8xf32>
    %4921 = stablehlo.convert %4920 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4922 = stablehlo.dot_general %4896, %4921, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4923 = stablehlo.transpose %4922, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4924 = stablehlo.reshape %4923 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4925 = sdy.sharding_constraint %4924 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4926 = stablehlo.reshape %4925 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4927 = stablehlo.convert %arg417 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4928 = stablehlo.dot_general %4926, %4927, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4929 = stablehlo.add %4778, %4928 : tensor<1x8x1024xbf16>
    %4930 = stablehlo.convert %4929 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4931 = stablehlo.multiply %4930, %4930 : tensor<1x8x1024xf32>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4932 = stablehlo.reduce(%4931 init: %cst_205) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4933 = stablehlo.broadcast_in_dim %4932, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4934 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4935 = stablehlo.divide %4933, %4934 : tensor<1x8x1xf32>
    %4936 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4937 = stablehlo.add %4935, %4936 : tensor<1x8x1xf32>
    %4938 = stablehlo.rsqrt %4937 : tensor<1x8x1xf32>
    %4939 = stablehlo.broadcast_in_dim %4938, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4940 = stablehlo.multiply %4930, %4939 : tensor<1x8x1024xf32>
    %4941 = stablehlo.convert %4940 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4942 = stablehlo.convert %arg414 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4943 = stablehlo.broadcast_in_dim %4942, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4944 = stablehlo.broadcast_in_dim %4943, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4945 = stablehlo.multiply %4944, %4941 : tensor<1x8x1024xbf16>
    %4946 = stablehlo.convert %arg412 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4947 = stablehlo.dot_general %4945, %4946, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4948 = call @silu(%4947) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4949 = stablehlo.convert %arg413 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4950 = stablehlo.dot_general %4945, %4949, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4951 = stablehlo.multiply %4948, %4950 : tensor<1x8x3072xbf16>
    %4952 = stablehlo.convert %arg411 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4953 = stablehlo.dot_general %4951, %4952, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4954 = stablehlo.add %4929, %4953 : tensor<1x8x1024xbf16>
    %4955 = stablehlo.convert %4954 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4956 = stablehlo.multiply %4955, %4955 : tensor<1x8x1024xf32>
    %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4957 = stablehlo.reduce(%4956 init: %cst_206) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4958 = stablehlo.broadcast_in_dim %4957, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4959 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4960 = stablehlo.divide %4958, %4959 : tensor<1x8x1xf32>
    %4961 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4962 = stablehlo.add %4960, %4961 : tensor<1x8x1xf32>
    %4963 = stablehlo.rsqrt %4962 : tensor<1x8x1xf32>
    %4964 = stablehlo.broadcast_in_dim %4963, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4965 = stablehlo.multiply %4955, %4964 : tensor<1x8x1024xf32>
    %4966 = stablehlo.convert %4965 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4967 = stablehlo.convert %arg421 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4968 = stablehlo.broadcast_in_dim %4967, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4969 = stablehlo.broadcast_in_dim %4968, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4970 = stablehlo.multiply %4969, %4966 : tensor<1x8x1024xbf16>
    %4971 = stablehlo.transpose %arg112, dims = [1, 0] : (tensor<151936x1024xf32>) -> tensor<1024x151936xf32>
    %4972 = stablehlo.convert %4971 : (tensor<1024x151936xf32>) -> tensor<1024x151936xbf16>
    %4973 = stablehlo.dot_general %4970, %4972, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x151936xbf16>) -> tensor<1x8x151936xbf16>
    return %4973 : tensor<1x8x151936xbf16>
  }
  func.func private @_take(%arg0: tensor<151936x1024xbf16>, %arg1: tensor<1x8xi32>) -> tensor<1x8x1024xbf16> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_0 = stablehlo.constant dense<151936> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x8xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1x8xi1>, tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %c_1 = stablehlo.constant dense<151935> : tensor<1xi32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x8x1xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<1x8x1xi32>, tensor<1x8x1xi32>) -> tensor<1x8x1xi1>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x8x1xi32>
    %10 = stablehlo.compare  LE, %5, %9,  SIGNED : (tensor<1x8x1xi32>, tensor<1x8x1xi32>) -> tensor<1x8x1xi1>
    %11 = stablehlo.and %7, %10 : tensor<1x8x1xi1>
    %c_3 = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.reduce(%11 init: %c_3) applies stablehlo.and across dimensions = [2] : (tensor<1x8x1xi1>, tensor<i1>) -> tensor<1x8xi1>
    %13 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<151936x1024xbf16>, tensor<1x8x1xi32>) -> tensor<1x8x1024xbf16>
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x8xi1>) -> tensor<1x8x1024xi1>
    %cst = stablehlo.constant dense<0x7FC0> : tensor<bf16>
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x8x1024xbf16>
    %16 = stablehlo.select %14, %13, %15 : tensor<1x8x1024xi1>, tensor<1x8x1024xbf16>
    return %16 : tensor<1x8x1024xbf16>
  }
  func.func private @_where(%arg0: tensor<1x8xi1>, %arg1: tensor<1x8xi32>, %arg2: tensor<1x8xi32>) -> tensor<1x8xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x8xi1>, tensor<1x8xi32>
    return %0 : tensor<1x8xi32>
  }
  func.func private @cumsum(%arg0: tensor<1x8xi1>) -> tensor<1x8xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x8xi1>) -> tensor<1x8xi32>
    %1 = call @cumsum_8(%0) : (tensor<1x8xi32>) -> tensor<1x8xi32>
    return %1 : tensor<1x8xi32>
  }
  func.func private @cumsum_8(%arg0: tensor<1x8xi32>) -> tensor<1x8xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [7, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 8>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<1x8xi32>, tensor<i32>) -> tensor<1x8xi32>
    return %1 : tensor<1x8xi32>
  }
  func.func private @clip(%arg0: tensor<1x8xi32>, %arg1: tensor<i32>) -> tensor<1x8xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x8xi32>
    return %2 : tensor<1x8xi32>
  }
  func.func private @get_frequencies() -> tensor<40960x128xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01, 3.200000e+01, 3.400000e+01, 3.600000e+01, 3.800000e+01, 4.000000e+01, 4.200000e+01, 4.400000e+01, 4.600000e+01, 4.800000e+01, 5.000000e+01, 5.200000e+01, 5.400000e+01, 5.600000e+01, 5.800000e+01, 6.000000e+01, 6.200000e+01, 6.400000e+01, 6.600000e+01, 6.800000e+01, 7.000000e+01, 7.200000e+01, 7.400000e+01, 7.600000e+01, 7.800000e+01, 8.000000e+01, 8.200000e+01, 8.400000e+01, 8.600000e+01, 8.800000e+01, 9.000000e+01, 9.200000e+01, 9.400000e+01, 9.600000e+01, 9.800000e+01, 1.000000e+02, 1.020000e+02, 1.040000e+02, 1.060000e+02, 1.080000e+02, 1.100000e+02, 1.120000e+02, 1.140000e+02, 1.160000e+02, 1.180000e+02, 1.200000e+02, 1.220000e+02, 1.240000e+02, 1.260000e+02]> : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.divide %cst, %0 : tensor<64xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+06> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.power %2, %1 : tensor<64xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5 = stablehlo.divide %4, %3 : tensor<64xf32>
    %6 = stablehlo.iota dim = 0 : tensor<40960xf32>
    %7 = stablehlo.dot_general %6, %5, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<40960xf32>, tensor<64xf32>) -> tensor<40960x64xf32>
    %8 = stablehlo.cosine %7 : tensor<40960x64xf32>
    %9 = stablehlo.sine %7 : tensor<40960x64xf32>
    %10 = stablehlo.concatenate %8, %9, dim = 1 : (tensor<40960x64xf32>, tensor<40960x64xf32>) -> tensor<40960x128xf32>
    return %10 : tensor<40960x128xf32>
  }
  func.func private @_where_91(%arg0: tensor<1x8x1x8x8xi1>, %arg1: tensor<1x8x2x8x8xbf16>, %arg2: tensor<bf16>) -> tensor<1x8x2x8x8xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3, 4] : (tensor<1x8x1x8x8xi1>) -> tensor<1x8x2x8x8xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<1x8x2x8x8xi1>, tensor<1x8x2x8x8xbf16>
    return %2 : tensor<1x8x2x8x8xbf16>
  }
  func.func private @silu(%arg0: tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16> {
    %0 = stablehlo.negate %arg0 : tensor<1x8x3072xbf16>
    %1 = stablehlo.exponential %0 : tensor<1x8x3072xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x8x3072xbf16>
    %3 = stablehlo.add %2, %1 : tensor<1x8x3072xbf16>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x8x3072xbf16>
    %5 = stablehlo.divide %4, %3 : tensor<1x8x3072xbf16>
    %6 = stablehlo.multiply %arg0, %5 : tensor<1x8x3072xbf16>
    return %6 : tensor<1x8x3072xbf16>
  }
}
