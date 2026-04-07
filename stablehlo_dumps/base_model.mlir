module @jit__forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["X"=1]>
  func.func public @main(%arg0: tensor<151936x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<3072x1024xf32>, %arg3: tensor<1024x3072xf32>, %arg4: tensor<1024x3072xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<128xf32>, %arg7: tensor<1024x1024xf32>, %arg8: tensor<2048x1024xf32>, %arg9: tensor<128xf32>, %arg10: tensor<1024x2048xf32>, %arg11: tensor<1024x1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<3072x1024xf32>, %arg14: tensor<1024x3072xf32>, %arg15: tensor<1024x3072xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<128xf32>, %arg18: tensor<1024x1024xf32>, %arg19: tensor<2048x1024xf32>, %arg20: tensor<128xf32>, %arg21: tensor<1024x2048xf32>, %arg22: tensor<1024x1024xf32>, %arg23: tensor<1024xf32>, %arg24: tensor<3072x1024xf32>, %arg25: tensor<1024x3072xf32>, %arg26: tensor<1024x3072xf32>, %arg27: tensor<1024xf32>, %arg28: tensor<128xf32>, %arg29: tensor<1024x1024xf32>, %arg30: tensor<2048x1024xf32>, %arg31: tensor<128xf32>, %arg32: tensor<1024x2048xf32>, %arg33: tensor<1024x1024xf32>, %arg34: tensor<1024xf32>, %arg35: tensor<3072x1024xf32>, %arg36: tensor<1024x3072xf32>, %arg37: tensor<1024x3072xf32>, %arg38: tensor<1024xf32>, %arg39: tensor<128xf32>, %arg40: tensor<1024x1024xf32>, %arg41: tensor<2048x1024xf32>, %arg42: tensor<128xf32>, %arg43: tensor<1024x2048xf32>, %arg44: tensor<1024x1024xf32>, %arg45: tensor<1024xf32>, %arg46: tensor<3072x1024xf32>, %arg47: tensor<1024x3072xf32>, %arg48: tensor<1024x3072xf32>, %arg49: tensor<1024xf32>, %arg50: tensor<128xf32>, %arg51: tensor<1024x1024xf32>, %arg52: tensor<2048x1024xf32>, %arg53: tensor<128xf32>, %arg54: tensor<1024x2048xf32>, %arg55: tensor<1024x1024xf32>, %arg56: tensor<1024xf32>, %arg57: tensor<3072x1024xf32>, %arg58: tensor<1024x3072xf32>, %arg59: tensor<1024x3072xf32>, %arg60: tensor<1024xf32>, %arg61: tensor<128xf32>, %arg62: tensor<1024x1024xf32>, %arg63: tensor<2048x1024xf32>, %arg64: tensor<128xf32>, %arg65: tensor<1024x2048xf32>, %arg66: tensor<1024x1024xf32>, %arg67: tensor<1024xf32>, %arg68: tensor<3072x1024xf32>, %arg69: tensor<1024x3072xf32>, %arg70: tensor<1024x3072xf32>, %arg71: tensor<1024xf32>, %arg72: tensor<128xf32>, %arg73: tensor<1024x1024xf32>, %arg74: tensor<2048x1024xf32>, %arg75: tensor<128xf32>, %arg76: tensor<1024x2048xf32>, %arg77: tensor<1024x1024xf32>, %arg78: tensor<1024xf32>, %arg79: tensor<3072x1024xf32>, %arg80: tensor<1024x3072xf32>, %arg81: tensor<1024x3072xf32>, %arg82: tensor<1024xf32>, %arg83: tensor<128xf32>, %arg84: tensor<1024x1024xf32>, %arg85: tensor<2048x1024xf32>, %arg86: tensor<128xf32>, %arg87: tensor<1024x2048xf32>, %arg88: tensor<1024x1024xf32>, %arg89: tensor<1024xf32>, %arg90: tensor<3072x1024xf32>, %arg91: tensor<1024x3072xf32>, %arg92: tensor<1024x3072xf32>, %arg93: tensor<1024xf32>, %arg94: tensor<128xf32>, %arg95: tensor<1024x1024xf32>, %arg96: tensor<2048x1024xf32>, %arg97: tensor<128xf32>, %arg98: tensor<1024x2048xf32>, %arg99: tensor<1024x1024xf32>, %arg100: tensor<1024xf32>, %arg101: tensor<3072x1024xf32>, %arg102: tensor<1024x3072xf32>, %arg103: tensor<1024x3072xf32>, %arg104: tensor<1024xf32>, %arg105: tensor<128xf32>, %arg106: tensor<1024x1024xf32>, %arg107: tensor<2048x1024xf32>, %arg108: tensor<128xf32>, %arg109: tensor<1024x2048xf32>, %arg110: tensor<1024x1024xf32>, %arg111: tensor<1024xf32>, %arg112: tensor<3072x1024xf32>, %arg113: tensor<1024x3072xf32>, %arg114: tensor<1024x3072xf32>, %arg115: tensor<1024xf32>, %arg116: tensor<128xf32>, %arg117: tensor<1024x1024xf32>, %arg118: tensor<2048x1024xf32>, %arg119: tensor<128xf32>, %arg120: tensor<1024x2048xf32>, %arg121: tensor<1024x1024xf32>, %arg122: tensor<1024xf32>, %arg123: tensor<3072x1024xf32>, %arg124: tensor<1024x3072xf32>, %arg125: tensor<1024x3072xf32>, %arg126: tensor<1024xf32>, %arg127: tensor<128xf32>, %arg128: tensor<1024x1024xf32>, %arg129: tensor<2048x1024xf32>, %arg130: tensor<128xf32>, %arg131: tensor<1024x2048xf32>, %arg132: tensor<1024x1024xf32>, %arg133: tensor<1024xf32>, %arg134: tensor<3072x1024xf32>, %arg135: tensor<1024x3072xf32>, %arg136: tensor<1024x3072xf32>, %arg137: tensor<1024xf32>, %arg138: tensor<128xf32>, %arg139: tensor<1024x1024xf32>, %arg140: tensor<2048x1024xf32>, %arg141: tensor<128xf32>, %arg142: tensor<1024x2048xf32>, %arg143: tensor<1024x1024xf32>, %arg144: tensor<1024xf32>, %arg145: tensor<3072x1024xf32>, %arg146: tensor<1024x3072xf32>, %arg147: tensor<1024x3072xf32>, %arg148: tensor<1024xf32>, %arg149: tensor<128xf32>, %arg150: tensor<1024x1024xf32>, %arg151: tensor<2048x1024xf32>, %arg152: tensor<128xf32>, %arg153: tensor<1024x2048xf32>, %arg154: tensor<1024x1024xf32>, %arg155: tensor<1024xf32>, %arg156: tensor<3072x1024xf32>, %arg157: tensor<1024x3072xf32>, %arg158: tensor<1024x3072xf32>, %arg159: tensor<1024xf32>, %arg160: tensor<128xf32>, %arg161: tensor<1024x1024xf32>, %arg162: tensor<2048x1024xf32>, %arg163: tensor<128xf32>, %arg164: tensor<1024x2048xf32>, %arg165: tensor<1024x1024xf32>, %arg166: tensor<1024xf32>, %arg167: tensor<3072x1024xf32>, %arg168: tensor<1024x3072xf32>, %arg169: tensor<1024x3072xf32>, %arg170: tensor<1024xf32>, %arg171: tensor<128xf32>, %arg172: tensor<1024x1024xf32>, %arg173: tensor<2048x1024xf32>, %arg174: tensor<128xf32>, %arg175: tensor<1024x2048xf32>, %arg176: tensor<1024x1024xf32>, %arg177: tensor<1024xf32>, %arg178: tensor<3072x1024xf32>, %arg179: tensor<1024x3072xf32>, %arg180: tensor<1024x3072xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<128xf32>, %arg183: tensor<1024x1024xf32>, %arg184: tensor<2048x1024xf32>, %arg185: tensor<128xf32>, %arg186: tensor<1024x2048xf32>, %arg187: tensor<1024x1024xf32>, %arg188: tensor<1024xf32>, %arg189: tensor<3072x1024xf32>, %arg190: tensor<1024x3072xf32>, %arg191: tensor<1024x3072xf32>, %arg192: tensor<1024xf32>, %arg193: tensor<128xf32>, %arg194: tensor<1024x1024xf32>, %arg195: tensor<2048x1024xf32>, %arg196: tensor<128xf32>, %arg197: tensor<1024x2048xf32>, %arg198: tensor<1024x1024xf32>, %arg199: tensor<1024xf32>, %arg200: tensor<3072x1024xf32>, %arg201: tensor<1024x3072xf32>, %arg202: tensor<1024x3072xf32>, %arg203: tensor<1024xf32>, %arg204: tensor<128xf32>, %arg205: tensor<1024x1024xf32>, %arg206: tensor<2048x1024xf32>, %arg207: tensor<128xf32>, %arg208: tensor<1024x2048xf32>, %arg209: tensor<1024x1024xf32>, %arg210: tensor<1024xf32>, %arg211: tensor<3072x1024xf32>, %arg212: tensor<1024x3072xf32>, %arg213: tensor<1024x3072xf32>, %arg214: tensor<1024xf32>, %arg215: tensor<128xf32>, %arg216: tensor<1024x1024xf32>, %arg217: tensor<2048x1024xf32>, %arg218: tensor<128xf32>, %arg219: tensor<1024x2048xf32>, %arg220: tensor<1024x1024xf32>, %arg221: tensor<1024xf32>, %arg222: tensor<3072x1024xf32>, %arg223: tensor<1024x3072xf32>, %arg224: tensor<1024x3072xf32>, %arg225: tensor<1024xf32>, %arg226: tensor<128xf32>, %arg227: tensor<1024x1024xf32>, %arg228: tensor<2048x1024xf32>, %arg229: tensor<128xf32>, %arg230: tensor<1024x2048xf32>, %arg231: tensor<1024x1024xf32>, %arg232: tensor<1024xf32>, %arg233: tensor<3072x1024xf32>, %arg234: tensor<1024x3072xf32>, %arg235: tensor<1024x3072xf32>, %arg236: tensor<1024xf32>, %arg237: tensor<128xf32>, %arg238: tensor<1024x1024xf32>, %arg239: tensor<2048x1024xf32>, %arg240: tensor<128xf32>, %arg241: tensor<1024x2048xf32>, %arg242: tensor<1024x1024xf32>, %arg243: tensor<1024xf32>, %arg244: tensor<3072x1024xf32>, %arg245: tensor<1024x3072xf32>, %arg246: tensor<1024x3072xf32>, %arg247: tensor<1024xf32>, %arg248: tensor<128xf32>, %arg249: tensor<1024x1024xf32>, %arg250: tensor<2048x1024xf32>, %arg251: tensor<128xf32>, %arg252: tensor<1024x2048xf32>, %arg253: tensor<1024x1024xf32>, %arg254: tensor<1024xf32>, %arg255: tensor<3072x1024xf32>, %arg256: tensor<1024x3072xf32>, %arg257: tensor<1024x3072xf32>, %arg258: tensor<1024xf32>, %arg259: tensor<128xf32>, %arg260: tensor<1024x1024xf32>, %arg261: tensor<2048x1024xf32>, %arg262: tensor<128xf32>, %arg263: tensor<1024x2048xf32>, %arg264: tensor<1024x1024xf32>, %arg265: tensor<1024xf32>, %arg266: tensor<3072x1024xf32>, %arg267: tensor<1024x3072xf32>, %arg268: tensor<1024x3072xf32>, %arg269: tensor<1024xf32>, %arg270: tensor<128xf32>, %arg271: tensor<1024x1024xf32>, %arg272: tensor<2048x1024xf32>, %arg273: tensor<128xf32>, %arg274: tensor<1024x2048xf32>, %arg275: tensor<1024x1024xf32>, %arg276: tensor<1024xf32>, %arg277: tensor<3072x1024xf32>, %arg278: tensor<1024x3072xf32>, %arg279: tensor<1024x3072xf32>, %arg280: tensor<1024xf32>, %arg281: tensor<128xf32>, %arg282: tensor<1024x1024xf32>, %arg283: tensor<2048x1024xf32>, %arg284: tensor<128xf32>, %arg285: tensor<1024x2048xf32>, %arg286: tensor<1024x1024xf32>, %arg287: tensor<1024xf32>, %arg288: tensor<3072x1024xf32>, %arg289: tensor<1024x3072xf32>, %arg290: tensor<1024x3072xf32>, %arg291: tensor<1024xf32>, %arg292: tensor<128xf32>, %arg293: tensor<1024x1024xf32>, %arg294: tensor<2048x1024xf32>, %arg295: tensor<128xf32>, %arg296: tensor<1024x2048xf32>, %arg297: tensor<1024x1024xf32>, %arg298: tensor<1024xf32>, %arg299: tensor<3072x1024xf32>, %arg300: tensor<1024x3072xf32>, %arg301: tensor<1024x3072xf32>, %arg302: tensor<1024xf32>, %arg303: tensor<128xf32>, %arg304: tensor<1024x1024xf32>, %arg305: tensor<2048x1024xf32>, %arg306: tensor<128xf32>, %arg307: tensor<1024x2048xf32>, %arg308: tensor<1024x1024xf32>, %arg309: tensor<1024xf32>, %arg310: tensor<1x8xui32>) -> (tensor<1x8x151936xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg310 : (tensor<1x8xui32>) -> tensor<1x8xi32>
    %1 = stablehlo.convert %arg0 : (tensor<151936x1024xf32>) -> tensor<151936x1024xbf16>
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
    %39 = stablehlo.convert %arg1 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %40 = stablehlo.broadcast_in_dim %39, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %42 = stablehlo.multiply %41, %38 : tensor<1x8x1024xbf16>
    %43 = stablehlo.convert %arg10 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %44 = stablehlo.dot_general %42, %43, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %45 = stablehlo.convert %arg7 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %46 = stablehlo.dot_general %42, %45, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %47 = stablehlo.convert %arg11 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %48 = stablehlo.dot_general %42, %47, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %49 = stablehlo.reshape %44 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %51 = stablehlo.multiply %50, %50 : tensor<1x8x16x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %52 = stablehlo.reduce(%51 init: %cst_5) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %cst_6 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %55 = stablehlo.divide %53, %54 : tensor<1x8x16x1xf32>
    %56 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x8x16x1xf32>
    %58 = stablehlo.rsqrt %57 : tensor<1x8x16x1xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %60 = stablehlo.multiply %50, %59 : tensor<1x8x16x128xf32>
    %61 = stablehlo.convert %60 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %62 = stablehlo.convert %arg9 : (tensor<128xf32>) -> tensor<128xbf16>
    %63 = stablehlo.broadcast_in_dim %62, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %65 = stablehlo.multiply %64, %61 : tensor<1x8x16x128xbf16>
    %66 = stablehlo.reshape %46 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %67 = stablehlo.convert %66 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %68 = stablehlo.multiply %67, %67 : tensor<1x8x8x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %69 = stablehlo.reduce(%68 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %71 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %72 = stablehlo.divide %70, %71 : tensor<1x8x8x1xf32>
    %73 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x8x8x1xf32>
    %75 = stablehlo.rsqrt %74 : tensor<1x8x8x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %77 = stablehlo.multiply %67, %76 : tensor<1x8x8x128xf32>
    %78 = stablehlo.convert %77 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %79 = stablehlo.convert %arg6 : (tensor<128xf32>) -> tensor<128xbf16>
    %80 = stablehlo.broadcast_in_dim %79, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %82 = stablehlo.multiply %81, %78 : tensor<1x8x8x128xbf16>
    %83 = stablehlo.reshape %48 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %84 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %85 = stablehlo.compare  LT, %7, %84,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %c_9 = stablehlo.constant dense<40960> : tensor<i32>
    %86 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %87 = stablehlo.add %7, %86 : tensor<1x8xi32>
    %88 = stablehlo.select %85, %87, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %90 = "stablehlo.gather"(%26, %89) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %91 = stablehlo.slice %90 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %92 = stablehlo.slice %90 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %93 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %94 = stablehlo.convert %93 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %95 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %96 = stablehlo.convert %95 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %97 = stablehlo.slice %65 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %98 = stablehlo.slice %65 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %99 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %100 = stablehlo.multiply %97, %99 : tensor<1x8x16x64xbf16>
    %101 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %102 = stablehlo.multiply %98, %101 : tensor<1x8x16x64xbf16>
    %103 = stablehlo.subtract %100, %102 : tensor<1x8x16x64xbf16>
    %104 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %105 = stablehlo.multiply %98, %104 : tensor<1x8x16x64xbf16>
    %106 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %107 = stablehlo.multiply %97, %106 : tensor<1x8x16x64xbf16>
    %108 = stablehlo.add %105, %107 : tensor<1x8x16x64xbf16>
    %109 = stablehlo.concatenate %103, %108, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %110 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %111 = stablehlo.convert %110 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %112 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %113 = stablehlo.convert %112 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %114 = stablehlo.slice %82 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %115 = stablehlo.slice %82 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %116 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %117 = stablehlo.multiply %114, %116 : tensor<1x8x8x64xbf16>
    %118 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %119 = stablehlo.multiply %115, %118 : tensor<1x8x8x64xbf16>
    %120 = stablehlo.subtract %117, %119 : tensor<1x8x8x64xbf16>
    %121 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %122 = stablehlo.multiply %115, %121 : tensor<1x8x8x64xbf16>
    %123 = stablehlo.broadcast_in_dim %113, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %124 = stablehlo.multiply %114, %123 : tensor<1x8x8x64xbf16>
    %125 = stablehlo.add %122, %124 : tensor<1x8x8x64xbf16>
    %126 = stablehlo.concatenate %120, %125, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %127 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %128 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %129 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %130 = stablehlo.and %129, %128 : tensor<1x1x8x8xi1>
    %131 = stablehlo.convert %130 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %132 = sdy.sharding_constraint %109 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %133 = sdy.sharding_constraint %126 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %134 = sdy.sharding_constraint %83 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %135 = sdy.sharding_constraint %131 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %136 = stablehlo.reshape %132 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %cst_10 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
    %137 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %138 = stablehlo.multiply %136, %137 : tensor<1x8x8x2x128xbf16>
    %139 = stablehlo.dot_general %133, %138, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %140 = stablehlo.transpose %139, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %141 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %142 = stablehlo.compare  NE, %135, %141,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %143 = stablehlo.convert %142 : tensor<1x1x8x8xi1>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %145 = stablehlo.reshape %144 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %cst_12 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
    %146 = call @_where_83(%145, %140, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %147 = stablehlo.convert %146 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_13 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %148 = stablehlo.reduce(%147 init: %cst_13) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %cst_14 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %149 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %150 = stablehlo.maximum %149, %148 : tensor<1x8x2x8xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %153 = stablehlo.subtract %147, %152 : tensor<1x8x2x8x8xf32>
    %154 = stablehlo.exponential %153 : tensor<1x8x2x8x8xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %155 = stablehlo.reduce(%154 init: %cst_15) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %158 = stablehlo.divide %154, %157 : tensor<1x8x2x8x8xf32>
    %159 = stablehlo.convert %158 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %160 = stablehlo.dot_general %134, %159, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %161 = stablehlo.transpose %160, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %162 = stablehlo.reshape %161 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %163 = sdy.sharding_constraint %162 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %164 = stablehlo.reshape %163 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %165 = stablehlo.convert %arg8 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %166 = stablehlo.dot_general %164, %165, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %167 = stablehlo.add %2, %166 : tensor<1x8x1024xbf16>
    %168 = stablehlo.convert %167 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %169 = stablehlo.multiply %168, %168 : tensor<1x8x1024xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.reduce(%169 init: %cst_16) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %172 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %173 = stablehlo.divide %171, %172 : tensor<1x8x1xf32>
    %174 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %175 = stablehlo.add %173, %174 : tensor<1x8x1xf32>
    %176 = stablehlo.rsqrt %175 : tensor<1x8x1xf32>
    %177 = stablehlo.broadcast_in_dim %176, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %178 = stablehlo.multiply %168, %177 : tensor<1x8x1024xf32>
    %179 = stablehlo.convert %178 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %180 = stablehlo.convert %arg5 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %181 = stablehlo.broadcast_in_dim %180, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %183 = stablehlo.multiply %182, %179 : tensor<1x8x1024xbf16>
    %184 = stablehlo.convert %arg3 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %185 = stablehlo.dot_general %183, %184, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %186 = call @silu(%185) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %187 = stablehlo.convert %arg4 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %188 = stablehlo.dot_general %183, %187, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %189 = stablehlo.multiply %186, %188 : tensor<1x8x3072xbf16>
    %190 = stablehlo.convert %arg2 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %191 = stablehlo.dot_general %189, %190, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %192 = stablehlo.add %167, %191 : tensor<1x8x1024xbf16>
    %193 = stablehlo.convert %192 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %194 = stablehlo.multiply %193, %193 : tensor<1x8x1024xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %195 = stablehlo.reduce(%194 init: %cst_17) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %197 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %198 = stablehlo.divide %196, %197 : tensor<1x8x1xf32>
    %199 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %200 = stablehlo.add %198, %199 : tensor<1x8x1xf32>
    %201 = stablehlo.rsqrt %200 : tensor<1x8x1xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %203 = stablehlo.multiply %193, %202 : tensor<1x8x1024xf32>
    %204 = stablehlo.convert %203 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %205 = stablehlo.convert %arg12 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %206 = stablehlo.broadcast_in_dim %205, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %208 = stablehlo.multiply %207, %204 : tensor<1x8x1024xbf16>
    %209 = stablehlo.convert %arg21 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %210 = stablehlo.dot_general %208, %209, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %211 = stablehlo.convert %arg18 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %212 = stablehlo.dot_general %208, %211, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %213 = stablehlo.convert %arg22 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %214 = stablehlo.dot_general %208, %213, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %215 = stablehlo.reshape %210 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %216 = stablehlo.convert %215 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %217 = stablehlo.multiply %216, %216 : tensor<1x8x16x128xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %218 = stablehlo.reduce(%217 init: %cst_18) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %220 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %221 = stablehlo.divide %219, %220 : tensor<1x8x16x1xf32>
    %222 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %223 = stablehlo.add %221, %222 : tensor<1x8x16x1xf32>
    %224 = stablehlo.rsqrt %223 : tensor<1x8x16x1xf32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %226 = stablehlo.multiply %216, %225 : tensor<1x8x16x128xf32>
    %227 = stablehlo.convert %226 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %228 = stablehlo.convert %arg20 : (tensor<128xf32>) -> tensor<128xbf16>
    %229 = stablehlo.broadcast_in_dim %228, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %231 = stablehlo.multiply %230, %227 : tensor<1x8x16x128xbf16>
    %232 = stablehlo.reshape %212 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %233 = stablehlo.convert %232 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %234 = stablehlo.multiply %233, %233 : tensor<1x8x8x128xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %235 = stablehlo.reduce(%234 init: %cst_19) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %236 = stablehlo.broadcast_in_dim %235, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %237 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %238 = stablehlo.divide %236, %237 : tensor<1x8x8x1xf32>
    %239 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %240 = stablehlo.add %238, %239 : tensor<1x8x8x1xf32>
    %241 = stablehlo.rsqrt %240 : tensor<1x8x8x1xf32>
    %242 = stablehlo.broadcast_in_dim %241, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %243 = stablehlo.multiply %233, %242 : tensor<1x8x8x128xf32>
    %244 = stablehlo.convert %243 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %245 = stablehlo.convert %arg17 : (tensor<128xf32>) -> tensor<128xbf16>
    %246 = stablehlo.broadcast_in_dim %245, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %248 = stablehlo.multiply %247, %244 : tensor<1x8x8x128xbf16>
    %249 = stablehlo.reshape %214 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %250 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %251 = stablehlo.compare  LT, %7, %250,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %252 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %253 = stablehlo.add %7, %252 : tensor<1x8xi32>
    %254 = stablehlo.select %251, %253, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %256 = "stablehlo.gather"(%26, %255) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %257 = stablehlo.slice %256 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %258 = stablehlo.slice %256 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %259 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %260 = stablehlo.convert %259 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %261 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %262 = stablehlo.convert %261 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %263 = stablehlo.slice %231 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %264 = stablehlo.slice %231 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %265 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %266 = stablehlo.multiply %263, %265 : tensor<1x8x16x64xbf16>
    %267 = stablehlo.broadcast_in_dim %262, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %268 = stablehlo.multiply %264, %267 : tensor<1x8x16x64xbf16>
    %269 = stablehlo.subtract %266, %268 : tensor<1x8x16x64xbf16>
    %270 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %271 = stablehlo.multiply %264, %270 : tensor<1x8x16x64xbf16>
    %272 = stablehlo.broadcast_in_dim %262, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %273 = stablehlo.multiply %263, %272 : tensor<1x8x16x64xbf16>
    %274 = stablehlo.add %271, %273 : tensor<1x8x16x64xbf16>
    %275 = stablehlo.concatenate %269, %274, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %276 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %277 = stablehlo.convert %276 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %278 = stablehlo.broadcast_in_dim %258, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %279 = stablehlo.convert %278 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %280 = stablehlo.slice %248 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %281 = stablehlo.slice %248 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %282 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %283 = stablehlo.multiply %280, %282 : tensor<1x8x8x64xbf16>
    %284 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %285 = stablehlo.multiply %281, %284 : tensor<1x8x8x64xbf16>
    %286 = stablehlo.subtract %283, %285 : tensor<1x8x8x64xbf16>
    %287 = stablehlo.broadcast_in_dim %277, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %288 = stablehlo.multiply %281, %287 : tensor<1x8x8x64xbf16>
    %289 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %290 = stablehlo.multiply %280, %289 : tensor<1x8x8x64xbf16>
    %291 = stablehlo.add %288, %290 : tensor<1x8x8x64xbf16>
    %292 = stablehlo.concatenate %286, %291, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %293 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %294 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %295 = stablehlo.broadcast_in_dim %293, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %296 = stablehlo.and %295, %294 : tensor<1x1x8x8xi1>
    %297 = stablehlo.convert %296 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %298 = sdy.sharding_constraint %275 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %299 = sdy.sharding_constraint %292 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %300 = sdy.sharding_constraint %249 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %301 = sdy.sharding_constraint %297 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %302 = stablehlo.reshape %298 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %303 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %304 = stablehlo.multiply %302, %303 : tensor<1x8x8x2x128xbf16>
    %305 = stablehlo.dot_general %299, %304, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %306 = stablehlo.transpose %305, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %307 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %308 = stablehlo.compare  NE, %301, %307,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %309 = stablehlo.convert %308 : tensor<1x1x8x8xi1>
    %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %311 = stablehlo.reshape %310 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %312 = call @_where_83(%311, %306, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %313 = stablehlo.convert %312 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %314 = stablehlo.reduce(%313 init: %cst_21) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %315 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %316 = stablehlo.maximum %315, %314 : tensor<1x8x2x8xf32>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %319 = stablehlo.subtract %313, %318 : tensor<1x8x2x8x8xf32>
    %320 = stablehlo.exponential %319 : tensor<1x8x2x8x8xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %321 = stablehlo.reduce(%320 init: %cst_22) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %324 = stablehlo.divide %320, %323 : tensor<1x8x2x8x8xf32>
    %325 = stablehlo.convert %324 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %326 = stablehlo.dot_general %300, %325, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %327 = stablehlo.transpose %326, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %328 = stablehlo.reshape %327 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %329 = sdy.sharding_constraint %328 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %330 = stablehlo.reshape %329 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %331 = stablehlo.convert %arg19 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %332 = stablehlo.dot_general %330, %331, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %333 = stablehlo.add %192, %332 : tensor<1x8x1024xbf16>
    %334 = stablehlo.convert %333 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %335 = stablehlo.multiply %334, %334 : tensor<1x8x1024xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %336 = stablehlo.reduce(%335 init: %cst_23) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %338 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %339 = stablehlo.divide %337, %338 : tensor<1x8x1xf32>
    %340 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %341 = stablehlo.add %339, %340 : tensor<1x8x1xf32>
    %342 = stablehlo.rsqrt %341 : tensor<1x8x1xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %344 = stablehlo.multiply %334, %343 : tensor<1x8x1024xf32>
    %345 = stablehlo.convert %344 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %346 = stablehlo.convert %arg16 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %347 = stablehlo.broadcast_in_dim %346, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %349 = stablehlo.multiply %348, %345 : tensor<1x8x1024xbf16>
    %350 = stablehlo.convert %arg14 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %351 = stablehlo.dot_general %349, %350, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %352 = call @silu(%351) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %353 = stablehlo.convert %arg15 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %354 = stablehlo.dot_general %349, %353, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %355 = stablehlo.multiply %352, %354 : tensor<1x8x3072xbf16>
    %356 = stablehlo.convert %arg13 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %357 = stablehlo.dot_general %355, %356, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %358 = stablehlo.add %333, %357 : tensor<1x8x1024xbf16>
    %359 = stablehlo.convert %358 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %360 = stablehlo.multiply %359, %359 : tensor<1x8x1024xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %361 = stablehlo.reduce(%360 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %363 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %364 = stablehlo.divide %362, %363 : tensor<1x8x1xf32>
    %365 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %366 = stablehlo.add %364, %365 : tensor<1x8x1xf32>
    %367 = stablehlo.rsqrt %366 : tensor<1x8x1xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %369 = stablehlo.multiply %359, %368 : tensor<1x8x1024xf32>
    %370 = stablehlo.convert %369 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %371 = stablehlo.convert %arg23 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %372 = stablehlo.broadcast_in_dim %371, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %374 = stablehlo.multiply %373, %370 : tensor<1x8x1024xbf16>
    %375 = stablehlo.convert %arg32 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %376 = stablehlo.dot_general %374, %375, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %377 = stablehlo.convert %arg29 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %378 = stablehlo.dot_general %374, %377, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %379 = stablehlo.convert %arg33 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %380 = stablehlo.dot_general %374, %379, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %381 = stablehlo.reshape %376 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %382 = stablehlo.convert %381 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %383 = stablehlo.multiply %382, %382 : tensor<1x8x16x128xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %384 = stablehlo.reduce(%383 init: %cst_25) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %386 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %387 = stablehlo.divide %385, %386 : tensor<1x8x16x1xf32>
    %388 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %389 = stablehlo.add %387, %388 : tensor<1x8x16x1xf32>
    %390 = stablehlo.rsqrt %389 : tensor<1x8x16x1xf32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %392 = stablehlo.multiply %382, %391 : tensor<1x8x16x128xf32>
    %393 = stablehlo.convert %392 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %394 = stablehlo.convert %arg31 : (tensor<128xf32>) -> tensor<128xbf16>
    %395 = stablehlo.broadcast_in_dim %394, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %397 = stablehlo.multiply %396, %393 : tensor<1x8x16x128xbf16>
    %398 = stablehlo.reshape %378 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %399 = stablehlo.convert %398 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %400 = stablehlo.multiply %399, %399 : tensor<1x8x8x128xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %401 = stablehlo.reduce(%400 init: %cst_26) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %403 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %404 = stablehlo.divide %402, %403 : tensor<1x8x8x1xf32>
    %405 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %406 = stablehlo.add %404, %405 : tensor<1x8x8x1xf32>
    %407 = stablehlo.rsqrt %406 : tensor<1x8x8x1xf32>
    %408 = stablehlo.broadcast_in_dim %407, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %409 = stablehlo.multiply %399, %408 : tensor<1x8x8x128xf32>
    %410 = stablehlo.convert %409 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %411 = stablehlo.convert %arg28 : (tensor<128xf32>) -> tensor<128xbf16>
    %412 = stablehlo.broadcast_in_dim %411, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %414 = stablehlo.multiply %413, %410 : tensor<1x8x8x128xbf16>
    %415 = stablehlo.reshape %380 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %416 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %417 = stablehlo.compare  LT, %7, %416,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %418 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %419 = stablehlo.add %7, %418 : tensor<1x8xi32>
    %420 = stablehlo.select %417, %419, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %421 = stablehlo.broadcast_in_dim %420, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %422 = "stablehlo.gather"(%26, %421) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %423 = stablehlo.slice %422 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %424 = stablehlo.slice %422 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %425 = stablehlo.broadcast_in_dim %423, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %426 = stablehlo.convert %425 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %427 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %428 = stablehlo.convert %427 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %429 = stablehlo.slice %397 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %430 = stablehlo.slice %397 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %431 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %432 = stablehlo.multiply %429, %431 : tensor<1x8x16x64xbf16>
    %433 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %434 = stablehlo.multiply %430, %433 : tensor<1x8x16x64xbf16>
    %435 = stablehlo.subtract %432, %434 : tensor<1x8x16x64xbf16>
    %436 = stablehlo.broadcast_in_dim %426, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %437 = stablehlo.multiply %430, %436 : tensor<1x8x16x64xbf16>
    %438 = stablehlo.broadcast_in_dim %428, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %439 = stablehlo.multiply %429, %438 : tensor<1x8x16x64xbf16>
    %440 = stablehlo.add %437, %439 : tensor<1x8x16x64xbf16>
    %441 = stablehlo.concatenate %435, %440, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %442 = stablehlo.broadcast_in_dim %423, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %443 = stablehlo.convert %442 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %444 = stablehlo.broadcast_in_dim %424, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %445 = stablehlo.convert %444 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %446 = stablehlo.slice %414 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %447 = stablehlo.slice %414 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %448 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %449 = stablehlo.multiply %446, %448 : tensor<1x8x8x64xbf16>
    %450 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %451 = stablehlo.multiply %447, %450 : tensor<1x8x8x64xbf16>
    %452 = stablehlo.subtract %449, %451 : tensor<1x8x8x64xbf16>
    %453 = stablehlo.broadcast_in_dim %443, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %454 = stablehlo.multiply %447, %453 : tensor<1x8x8x64xbf16>
    %455 = stablehlo.broadcast_in_dim %445, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %456 = stablehlo.multiply %446, %455 : tensor<1x8x8x64xbf16>
    %457 = stablehlo.add %454, %456 : tensor<1x8x8x64xbf16>
    %458 = stablehlo.concatenate %452, %457, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %459 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %460 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %461 = stablehlo.broadcast_in_dim %459, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %462 = stablehlo.and %461, %460 : tensor<1x1x8x8xi1>
    %463 = stablehlo.convert %462 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %464 = sdy.sharding_constraint %441 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %465 = sdy.sharding_constraint %458 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %466 = sdy.sharding_constraint %415 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %467 = sdy.sharding_constraint %463 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %468 = stablehlo.reshape %464 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %469 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %470 = stablehlo.multiply %468, %469 : tensor<1x8x8x2x128xbf16>
    %471 = stablehlo.dot_general %465, %470, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %472 = stablehlo.transpose %471, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %473 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %474 = stablehlo.compare  NE, %467, %473,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %475 = stablehlo.convert %474 : tensor<1x1x8x8xi1>
    %476 = stablehlo.broadcast_in_dim %475, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %477 = stablehlo.reshape %476 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %478 = call @_where_83(%477, %472, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %479 = stablehlo.convert %478 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %480 = stablehlo.reduce(%479 init: %cst_28) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %481 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %482 = stablehlo.maximum %481, %480 : tensor<1x8x2x8xf32>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %484 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %485 = stablehlo.subtract %479, %484 : tensor<1x8x2x8x8xf32>
    %486 = stablehlo.exponential %485 : tensor<1x8x2x8x8xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %487 = stablehlo.reduce(%486 init: %cst_29) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %488 = stablehlo.broadcast_in_dim %487, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %490 = stablehlo.divide %486, %489 : tensor<1x8x2x8x8xf32>
    %491 = stablehlo.convert %490 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %492 = stablehlo.dot_general %466, %491, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %493 = stablehlo.transpose %492, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %494 = stablehlo.reshape %493 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %495 = sdy.sharding_constraint %494 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %496 = stablehlo.reshape %495 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %497 = stablehlo.convert %arg30 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %498 = stablehlo.dot_general %496, %497, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %499 = stablehlo.add %358, %498 : tensor<1x8x1024xbf16>
    %500 = stablehlo.convert %499 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %501 = stablehlo.multiply %500, %500 : tensor<1x8x1024xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %502 = stablehlo.reduce(%501 init: %cst_30) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %503 = stablehlo.broadcast_in_dim %502, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %504 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %505 = stablehlo.divide %503, %504 : tensor<1x8x1xf32>
    %506 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %507 = stablehlo.add %505, %506 : tensor<1x8x1xf32>
    %508 = stablehlo.rsqrt %507 : tensor<1x8x1xf32>
    %509 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %510 = stablehlo.multiply %500, %509 : tensor<1x8x1024xf32>
    %511 = stablehlo.convert %510 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %512 = stablehlo.convert %arg27 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %513 = stablehlo.broadcast_in_dim %512, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %514 = stablehlo.broadcast_in_dim %513, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %515 = stablehlo.multiply %514, %511 : tensor<1x8x1024xbf16>
    %516 = stablehlo.convert %arg25 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %517 = stablehlo.dot_general %515, %516, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %518 = call @silu(%517) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %519 = stablehlo.convert %arg26 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %520 = stablehlo.dot_general %515, %519, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %521 = stablehlo.multiply %518, %520 : tensor<1x8x3072xbf16>
    %522 = stablehlo.convert %arg24 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %523 = stablehlo.dot_general %521, %522, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %524 = stablehlo.add %499, %523 : tensor<1x8x1024xbf16>
    %525 = stablehlo.convert %524 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %526 = stablehlo.multiply %525, %525 : tensor<1x8x1024xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %527 = stablehlo.reduce(%526 init: %cst_31) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %528 = stablehlo.broadcast_in_dim %527, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %529 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %530 = stablehlo.divide %528, %529 : tensor<1x8x1xf32>
    %531 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %532 = stablehlo.add %530, %531 : tensor<1x8x1xf32>
    %533 = stablehlo.rsqrt %532 : tensor<1x8x1xf32>
    %534 = stablehlo.broadcast_in_dim %533, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %535 = stablehlo.multiply %525, %534 : tensor<1x8x1024xf32>
    %536 = stablehlo.convert %535 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %537 = stablehlo.convert %arg34 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %538 = stablehlo.broadcast_in_dim %537, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %540 = stablehlo.multiply %539, %536 : tensor<1x8x1024xbf16>
    %541 = stablehlo.convert %arg43 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %542 = stablehlo.dot_general %540, %541, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %543 = stablehlo.convert %arg40 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %544 = stablehlo.dot_general %540, %543, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %545 = stablehlo.convert %arg44 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %546 = stablehlo.dot_general %540, %545, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %547 = stablehlo.reshape %542 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %548 = stablehlo.convert %547 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %549 = stablehlo.multiply %548, %548 : tensor<1x8x16x128xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %550 = stablehlo.reduce(%549 init: %cst_32) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %551 = stablehlo.broadcast_in_dim %550, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %552 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %553 = stablehlo.divide %551, %552 : tensor<1x8x16x1xf32>
    %554 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %555 = stablehlo.add %553, %554 : tensor<1x8x16x1xf32>
    %556 = stablehlo.rsqrt %555 : tensor<1x8x16x1xf32>
    %557 = stablehlo.broadcast_in_dim %556, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %558 = stablehlo.multiply %548, %557 : tensor<1x8x16x128xf32>
    %559 = stablehlo.convert %558 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %560 = stablehlo.convert %arg42 : (tensor<128xf32>) -> tensor<128xbf16>
    %561 = stablehlo.broadcast_in_dim %560, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %562 = stablehlo.broadcast_in_dim %561, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %563 = stablehlo.multiply %562, %559 : tensor<1x8x16x128xbf16>
    %564 = stablehlo.reshape %544 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %565 = stablehlo.convert %564 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %566 = stablehlo.multiply %565, %565 : tensor<1x8x8x128xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %567 = stablehlo.reduce(%566 init: %cst_33) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %569 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %570 = stablehlo.divide %568, %569 : tensor<1x8x8x1xf32>
    %571 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %572 = stablehlo.add %570, %571 : tensor<1x8x8x1xf32>
    %573 = stablehlo.rsqrt %572 : tensor<1x8x8x1xf32>
    %574 = stablehlo.broadcast_in_dim %573, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %575 = stablehlo.multiply %565, %574 : tensor<1x8x8x128xf32>
    %576 = stablehlo.convert %575 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %577 = stablehlo.convert %arg39 : (tensor<128xf32>) -> tensor<128xbf16>
    %578 = stablehlo.broadcast_in_dim %577, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %579 = stablehlo.broadcast_in_dim %578, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %580 = stablehlo.multiply %579, %576 : tensor<1x8x8x128xbf16>
    %581 = stablehlo.reshape %546 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %582 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %583 = stablehlo.compare  LT, %7, %582,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %584 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %585 = stablehlo.add %7, %584 : tensor<1x8xi32>
    %586 = stablehlo.select %583, %585, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %587 = stablehlo.broadcast_in_dim %586, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %588 = "stablehlo.gather"(%26, %587) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %589 = stablehlo.slice %588 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %590 = stablehlo.slice %588 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %591 = stablehlo.broadcast_in_dim %589, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %592 = stablehlo.convert %591 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %593 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %594 = stablehlo.convert %593 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %595 = stablehlo.slice %563 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %596 = stablehlo.slice %563 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %597 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %598 = stablehlo.multiply %595, %597 : tensor<1x8x16x64xbf16>
    %599 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %600 = stablehlo.multiply %596, %599 : tensor<1x8x16x64xbf16>
    %601 = stablehlo.subtract %598, %600 : tensor<1x8x16x64xbf16>
    %602 = stablehlo.broadcast_in_dim %592, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %603 = stablehlo.multiply %596, %602 : tensor<1x8x16x64xbf16>
    %604 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %605 = stablehlo.multiply %595, %604 : tensor<1x8x16x64xbf16>
    %606 = stablehlo.add %603, %605 : tensor<1x8x16x64xbf16>
    %607 = stablehlo.concatenate %601, %606, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %608 = stablehlo.broadcast_in_dim %589, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %609 = stablehlo.convert %608 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %610 = stablehlo.broadcast_in_dim %590, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %611 = stablehlo.convert %610 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %612 = stablehlo.slice %580 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %613 = stablehlo.slice %580 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %614 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %615 = stablehlo.multiply %612, %614 : tensor<1x8x8x64xbf16>
    %616 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %617 = stablehlo.multiply %613, %616 : tensor<1x8x8x64xbf16>
    %618 = stablehlo.subtract %615, %617 : tensor<1x8x8x64xbf16>
    %619 = stablehlo.broadcast_in_dim %609, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %620 = stablehlo.multiply %613, %619 : tensor<1x8x8x64xbf16>
    %621 = stablehlo.broadcast_in_dim %611, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %622 = stablehlo.multiply %612, %621 : tensor<1x8x8x64xbf16>
    %623 = stablehlo.add %620, %622 : tensor<1x8x8x64xbf16>
    %624 = stablehlo.concatenate %618, %623, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %625 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %626 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %627 = stablehlo.broadcast_in_dim %625, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %628 = stablehlo.and %627, %626 : tensor<1x1x8x8xi1>
    %629 = stablehlo.convert %628 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %630 = sdy.sharding_constraint %607 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %631 = sdy.sharding_constraint %624 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %632 = sdy.sharding_constraint %581 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %633 = sdy.sharding_constraint %629 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %634 = stablehlo.reshape %630 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %635 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %636 = stablehlo.multiply %634, %635 : tensor<1x8x8x2x128xbf16>
    %637 = stablehlo.dot_general %631, %636, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %638 = stablehlo.transpose %637, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %639 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %640 = stablehlo.compare  NE, %633, %639,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %641 = stablehlo.convert %640 : tensor<1x1x8x8xi1>
    %642 = stablehlo.broadcast_in_dim %641, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %643 = stablehlo.reshape %642 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %644 = call @_where_83(%643, %638, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %645 = stablehlo.convert %644 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_35 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %646 = stablehlo.reduce(%645 init: %cst_35) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %647 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %648 = stablehlo.maximum %647, %646 : tensor<1x8x2x8xf32>
    %649 = stablehlo.broadcast_in_dim %648, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %650 = stablehlo.broadcast_in_dim %649, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %651 = stablehlo.subtract %645, %650 : tensor<1x8x2x8x8xf32>
    %652 = stablehlo.exponential %651 : tensor<1x8x2x8x8xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %653 = stablehlo.reduce(%652 init: %cst_36) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %654 = stablehlo.broadcast_in_dim %653, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %655 = stablehlo.broadcast_in_dim %654, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %656 = stablehlo.divide %652, %655 : tensor<1x8x2x8x8xf32>
    %657 = stablehlo.convert %656 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %658 = stablehlo.dot_general %632, %657, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %659 = stablehlo.transpose %658, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %660 = stablehlo.reshape %659 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %661 = sdy.sharding_constraint %660 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %662 = stablehlo.reshape %661 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %663 = stablehlo.convert %arg41 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %664 = stablehlo.dot_general %662, %663, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %665 = stablehlo.add %524, %664 : tensor<1x8x1024xbf16>
    %666 = stablehlo.convert %665 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %667 = stablehlo.multiply %666, %666 : tensor<1x8x1024xf32>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %668 = stablehlo.reduce(%667 init: %cst_37) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %669 = stablehlo.broadcast_in_dim %668, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %670 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %671 = stablehlo.divide %669, %670 : tensor<1x8x1xf32>
    %672 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %673 = stablehlo.add %671, %672 : tensor<1x8x1xf32>
    %674 = stablehlo.rsqrt %673 : tensor<1x8x1xf32>
    %675 = stablehlo.broadcast_in_dim %674, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %676 = stablehlo.multiply %666, %675 : tensor<1x8x1024xf32>
    %677 = stablehlo.convert %676 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %678 = stablehlo.convert %arg38 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %679 = stablehlo.broadcast_in_dim %678, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %680 = stablehlo.broadcast_in_dim %679, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %681 = stablehlo.multiply %680, %677 : tensor<1x8x1024xbf16>
    %682 = stablehlo.convert %arg36 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %683 = stablehlo.dot_general %681, %682, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %684 = call @silu(%683) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %685 = stablehlo.convert %arg37 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %686 = stablehlo.dot_general %681, %685, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %687 = stablehlo.multiply %684, %686 : tensor<1x8x3072xbf16>
    %688 = stablehlo.convert %arg35 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %689 = stablehlo.dot_general %687, %688, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %690 = stablehlo.add %665, %689 : tensor<1x8x1024xbf16>
    %691 = stablehlo.convert %690 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %692 = stablehlo.multiply %691, %691 : tensor<1x8x1024xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %693 = stablehlo.reduce(%692 init: %cst_38) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %695 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %696 = stablehlo.divide %694, %695 : tensor<1x8x1xf32>
    %697 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %698 = stablehlo.add %696, %697 : tensor<1x8x1xf32>
    %699 = stablehlo.rsqrt %698 : tensor<1x8x1xf32>
    %700 = stablehlo.broadcast_in_dim %699, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %701 = stablehlo.multiply %691, %700 : tensor<1x8x1024xf32>
    %702 = stablehlo.convert %701 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %703 = stablehlo.convert %arg45 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %704 = stablehlo.broadcast_in_dim %703, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %705 = stablehlo.broadcast_in_dim %704, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %706 = stablehlo.multiply %705, %702 : tensor<1x8x1024xbf16>
    %707 = stablehlo.convert %arg54 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %708 = stablehlo.dot_general %706, %707, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %709 = stablehlo.convert %arg51 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %710 = stablehlo.dot_general %706, %709, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %711 = stablehlo.convert %arg55 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %712 = stablehlo.dot_general %706, %711, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %713 = stablehlo.reshape %708 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %714 = stablehlo.convert %713 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %715 = stablehlo.multiply %714, %714 : tensor<1x8x16x128xf32>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %716 = stablehlo.reduce(%715 init: %cst_39) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %718 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %719 = stablehlo.divide %717, %718 : tensor<1x8x16x1xf32>
    %720 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %721 = stablehlo.add %719, %720 : tensor<1x8x16x1xf32>
    %722 = stablehlo.rsqrt %721 : tensor<1x8x16x1xf32>
    %723 = stablehlo.broadcast_in_dim %722, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %724 = stablehlo.multiply %714, %723 : tensor<1x8x16x128xf32>
    %725 = stablehlo.convert %724 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %726 = stablehlo.convert %arg53 : (tensor<128xf32>) -> tensor<128xbf16>
    %727 = stablehlo.broadcast_in_dim %726, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %728 = stablehlo.broadcast_in_dim %727, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %729 = stablehlo.multiply %728, %725 : tensor<1x8x16x128xbf16>
    %730 = stablehlo.reshape %710 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %731 = stablehlo.convert %730 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %732 = stablehlo.multiply %731, %731 : tensor<1x8x8x128xf32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %733 = stablehlo.reduce(%732 init: %cst_40) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %734 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %735 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %736 = stablehlo.divide %734, %735 : tensor<1x8x8x1xf32>
    %737 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %738 = stablehlo.add %736, %737 : tensor<1x8x8x1xf32>
    %739 = stablehlo.rsqrt %738 : tensor<1x8x8x1xf32>
    %740 = stablehlo.broadcast_in_dim %739, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %741 = stablehlo.multiply %731, %740 : tensor<1x8x8x128xf32>
    %742 = stablehlo.convert %741 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %743 = stablehlo.convert %arg50 : (tensor<128xf32>) -> tensor<128xbf16>
    %744 = stablehlo.broadcast_in_dim %743, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %745 = stablehlo.broadcast_in_dim %744, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %746 = stablehlo.multiply %745, %742 : tensor<1x8x8x128xbf16>
    %747 = stablehlo.reshape %712 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %748 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %749 = stablehlo.compare  LT, %7, %748,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %750 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %751 = stablehlo.add %7, %750 : tensor<1x8xi32>
    %752 = stablehlo.select %749, %751, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %753 = stablehlo.broadcast_in_dim %752, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %754 = "stablehlo.gather"(%26, %753) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %755 = stablehlo.slice %754 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %756 = stablehlo.slice %754 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %757 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %758 = stablehlo.convert %757 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %759 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %760 = stablehlo.convert %759 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %761 = stablehlo.slice %729 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %762 = stablehlo.slice %729 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %763 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %764 = stablehlo.multiply %761, %763 : tensor<1x8x16x64xbf16>
    %765 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %766 = stablehlo.multiply %762, %765 : tensor<1x8x16x64xbf16>
    %767 = stablehlo.subtract %764, %766 : tensor<1x8x16x64xbf16>
    %768 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %769 = stablehlo.multiply %762, %768 : tensor<1x8x16x64xbf16>
    %770 = stablehlo.broadcast_in_dim %760, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %771 = stablehlo.multiply %761, %770 : tensor<1x8x16x64xbf16>
    %772 = stablehlo.add %769, %771 : tensor<1x8x16x64xbf16>
    %773 = stablehlo.concatenate %767, %772, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %774 = stablehlo.broadcast_in_dim %755, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %775 = stablehlo.convert %774 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %776 = stablehlo.broadcast_in_dim %756, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %777 = stablehlo.convert %776 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %778 = stablehlo.slice %746 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %779 = stablehlo.slice %746 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %780 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %781 = stablehlo.multiply %778, %780 : tensor<1x8x8x64xbf16>
    %782 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %783 = stablehlo.multiply %779, %782 : tensor<1x8x8x64xbf16>
    %784 = stablehlo.subtract %781, %783 : tensor<1x8x8x64xbf16>
    %785 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %786 = stablehlo.multiply %779, %785 : tensor<1x8x8x64xbf16>
    %787 = stablehlo.broadcast_in_dim %777, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %788 = stablehlo.multiply %778, %787 : tensor<1x8x8x64xbf16>
    %789 = stablehlo.add %786, %788 : tensor<1x8x8x64xbf16>
    %790 = stablehlo.concatenate %784, %789, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %791 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %792 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %793 = stablehlo.broadcast_in_dim %791, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %794 = stablehlo.and %793, %792 : tensor<1x1x8x8xi1>
    %795 = stablehlo.convert %794 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %796 = sdy.sharding_constraint %773 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %797 = sdy.sharding_constraint %790 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %798 = sdy.sharding_constraint %747 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %799 = sdy.sharding_constraint %795 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %800 = stablehlo.reshape %796 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %801 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %802 = stablehlo.multiply %800, %801 : tensor<1x8x8x2x128xbf16>
    %803 = stablehlo.dot_general %797, %802, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %804 = stablehlo.transpose %803, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %805 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %806 = stablehlo.compare  NE, %799, %805,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %807 = stablehlo.convert %806 : tensor<1x1x8x8xi1>
    %808 = stablehlo.broadcast_in_dim %807, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %809 = stablehlo.reshape %808 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %810 = call @_where_83(%809, %804, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %811 = stablehlo.convert %810 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_42 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %812 = stablehlo.reduce(%811 init: %cst_42) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %813 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %814 = stablehlo.maximum %813, %812 : tensor<1x8x2x8xf32>
    %815 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %816 = stablehlo.broadcast_in_dim %815, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %817 = stablehlo.subtract %811, %816 : tensor<1x8x2x8x8xf32>
    %818 = stablehlo.exponential %817 : tensor<1x8x2x8x8xf32>
    %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %819 = stablehlo.reduce(%818 init: %cst_43) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %820 = stablehlo.broadcast_in_dim %819, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %822 = stablehlo.divide %818, %821 : tensor<1x8x2x8x8xf32>
    %823 = stablehlo.convert %822 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %824 = stablehlo.dot_general %798, %823, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %825 = stablehlo.transpose %824, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %826 = stablehlo.reshape %825 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %827 = sdy.sharding_constraint %826 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %828 = stablehlo.reshape %827 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %829 = stablehlo.convert %arg52 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %830 = stablehlo.dot_general %828, %829, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %831 = stablehlo.add %690, %830 : tensor<1x8x1024xbf16>
    %832 = stablehlo.convert %831 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %833 = stablehlo.multiply %832, %832 : tensor<1x8x1024xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %834 = stablehlo.reduce(%833 init: %cst_44) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %836 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %837 = stablehlo.divide %835, %836 : tensor<1x8x1xf32>
    %838 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %839 = stablehlo.add %837, %838 : tensor<1x8x1xf32>
    %840 = stablehlo.rsqrt %839 : tensor<1x8x1xf32>
    %841 = stablehlo.broadcast_in_dim %840, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %842 = stablehlo.multiply %832, %841 : tensor<1x8x1024xf32>
    %843 = stablehlo.convert %842 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %844 = stablehlo.convert %arg49 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %845 = stablehlo.broadcast_in_dim %844, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %846 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %847 = stablehlo.multiply %846, %843 : tensor<1x8x1024xbf16>
    %848 = stablehlo.convert %arg47 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %849 = stablehlo.dot_general %847, %848, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %850 = call @silu(%849) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %851 = stablehlo.convert %arg48 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %852 = stablehlo.dot_general %847, %851, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %853 = stablehlo.multiply %850, %852 : tensor<1x8x3072xbf16>
    %854 = stablehlo.convert %arg46 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %855 = stablehlo.dot_general %853, %854, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %856 = stablehlo.add %831, %855 : tensor<1x8x1024xbf16>
    %857 = stablehlo.convert %856 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %858 = stablehlo.multiply %857, %857 : tensor<1x8x1024xf32>
    %cst_45 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %859 = stablehlo.reduce(%858 init: %cst_45) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %860 = stablehlo.broadcast_in_dim %859, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %861 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %862 = stablehlo.divide %860, %861 : tensor<1x8x1xf32>
    %863 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %864 = stablehlo.add %862, %863 : tensor<1x8x1xf32>
    %865 = stablehlo.rsqrt %864 : tensor<1x8x1xf32>
    %866 = stablehlo.broadcast_in_dim %865, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %867 = stablehlo.multiply %857, %866 : tensor<1x8x1024xf32>
    %868 = stablehlo.convert %867 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %869 = stablehlo.convert %arg56 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %870 = stablehlo.broadcast_in_dim %869, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %871 = stablehlo.broadcast_in_dim %870, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %872 = stablehlo.multiply %871, %868 : tensor<1x8x1024xbf16>
    %873 = stablehlo.convert %arg65 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %874 = stablehlo.dot_general %872, %873, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %875 = stablehlo.convert %arg62 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %876 = stablehlo.dot_general %872, %875, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %877 = stablehlo.convert %arg66 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %878 = stablehlo.dot_general %872, %877, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %879 = stablehlo.reshape %874 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %880 = stablehlo.convert %879 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %881 = stablehlo.multiply %880, %880 : tensor<1x8x16x128xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %882 = stablehlo.reduce(%881 init: %cst_46) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %884 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %885 = stablehlo.divide %883, %884 : tensor<1x8x16x1xf32>
    %886 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %887 = stablehlo.add %885, %886 : tensor<1x8x16x1xf32>
    %888 = stablehlo.rsqrt %887 : tensor<1x8x16x1xf32>
    %889 = stablehlo.broadcast_in_dim %888, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %890 = stablehlo.multiply %880, %889 : tensor<1x8x16x128xf32>
    %891 = stablehlo.convert %890 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %892 = stablehlo.convert %arg64 : (tensor<128xf32>) -> tensor<128xbf16>
    %893 = stablehlo.broadcast_in_dim %892, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %894 = stablehlo.broadcast_in_dim %893, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %895 = stablehlo.multiply %894, %891 : tensor<1x8x16x128xbf16>
    %896 = stablehlo.reshape %876 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %897 = stablehlo.convert %896 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %898 = stablehlo.multiply %897, %897 : tensor<1x8x8x128xf32>
    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %899 = stablehlo.reduce(%898 init: %cst_47) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %900 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %901 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %902 = stablehlo.divide %900, %901 : tensor<1x8x8x1xf32>
    %903 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %904 = stablehlo.add %902, %903 : tensor<1x8x8x1xf32>
    %905 = stablehlo.rsqrt %904 : tensor<1x8x8x1xf32>
    %906 = stablehlo.broadcast_in_dim %905, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %907 = stablehlo.multiply %897, %906 : tensor<1x8x8x128xf32>
    %908 = stablehlo.convert %907 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %909 = stablehlo.convert %arg61 : (tensor<128xf32>) -> tensor<128xbf16>
    %910 = stablehlo.broadcast_in_dim %909, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %911 = stablehlo.broadcast_in_dim %910, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %912 = stablehlo.multiply %911, %908 : tensor<1x8x8x128xbf16>
    %913 = stablehlo.reshape %878 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %914 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %915 = stablehlo.compare  LT, %7, %914,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %916 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %917 = stablehlo.add %7, %916 : tensor<1x8xi32>
    %918 = stablehlo.select %915, %917, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %919 = stablehlo.broadcast_in_dim %918, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %920 = "stablehlo.gather"(%26, %919) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %921 = stablehlo.slice %920 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %922 = stablehlo.slice %920 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %923 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %924 = stablehlo.convert %923 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %925 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %926 = stablehlo.convert %925 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %927 = stablehlo.slice %895 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %928 = stablehlo.slice %895 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %929 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %930 = stablehlo.multiply %927, %929 : tensor<1x8x16x64xbf16>
    %931 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %932 = stablehlo.multiply %928, %931 : tensor<1x8x16x64xbf16>
    %933 = stablehlo.subtract %930, %932 : tensor<1x8x16x64xbf16>
    %934 = stablehlo.broadcast_in_dim %924, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %935 = stablehlo.multiply %928, %934 : tensor<1x8x16x64xbf16>
    %936 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %937 = stablehlo.multiply %927, %936 : tensor<1x8x16x64xbf16>
    %938 = stablehlo.add %935, %937 : tensor<1x8x16x64xbf16>
    %939 = stablehlo.concatenate %933, %938, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %940 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %941 = stablehlo.convert %940 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %942 = stablehlo.broadcast_in_dim %922, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %943 = stablehlo.convert %942 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %944 = stablehlo.slice %912 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %945 = stablehlo.slice %912 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %946 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %947 = stablehlo.multiply %944, %946 : tensor<1x8x8x64xbf16>
    %948 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %949 = stablehlo.multiply %945, %948 : tensor<1x8x8x64xbf16>
    %950 = stablehlo.subtract %947, %949 : tensor<1x8x8x64xbf16>
    %951 = stablehlo.broadcast_in_dim %941, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %952 = stablehlo.multiply %945, %951 : tensor<1x8x8x64xbf16>
    %953 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %954 = stablehlo.multiply %944, %953 : tensor<1x8x8x64xbf16>
    %955 = stablehlo.add %952, %954 : tensor<1x8x8x64xbf16>
    %956 = stablehlo.concatenate %950, %955, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %957 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %958 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %959 = stablehlo.broadcast_in_dim %957, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %960 = stablehlo.and %959, %958 : tensor<1x1x8x8xi1>
    %961 = stablehlo.convert %960 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %962 = sdy.sharding_constraint %939 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %963 = sdy.sharding_constraint %956 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %964 = sdy.sharding_constraint %913 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %965 = sdy.sharding_constraint %961 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %966 = stablehlo.reshape %962 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %967 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %968 = stablehlo.multiply %966, %967 : tensor<1x8x8x2x128xbf16>
    %969 = stablehlo.dot_general %963, %968, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %970 = stablehlo.transpose %969, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_48 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %971 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %972 = stablehlo.compare  NE, %965, %971,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %973 = stablehlo.convert %972 : tensor<1x1x8x8xi1>
    %974 = stablehlo.broadcast_in_dim %973, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %975 = stablehlo.reshape %974 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %976 = call @_where_83(%975, %970, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %977 = stablehlo.convert %976 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_49 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %978 = stablehlo.reduce(%977 init: %cst_49) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %979 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %980 = stablehlo.maximum %979, %978 : tensor<1x8x2x8xf32>
    %981 = stablehlo.broadcast_in_dim %980, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %983 = stablehlo.subtract %977, %982 : tensor<1x8x2x8x8xf32>
    %984 = stablehlo.exponential %983 : tensor<1x8x2x8x8xf32>
    %cst_50 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %985 = stablehlo.reduce(%984 init: %cst_50) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %986 = stablehlo.broadcast_in_dim %985, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %988 = stablehlo.divide %984, %987 : tensor<1x8x2x8x8xf32>
    %989 = stablehlo.convert %988 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %990 = stablehlo.dot_general %964, %989, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %991 = stablehlo.transpose %990, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %992 = stablehlo.reshape %991 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %993 = sdy.sharding_constraint %992 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %994 = stablehlo.reshape %993 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %995 = stablehlo.convert %arg63 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %996 = stablehlo.dot_general %994, %995, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %997 = stablehlo.add %856, %996 : tensor<1x8x1024xbf16>
    %998 = stablehlo.convert %997 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %999 = stablehlo.multiply %998, %998 : tensor<1x8x1024xf32>
    %cst_51 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1000 = stablehlo.reduce(%999 init: %cst_51) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1001 = stablehlo.broadcast_in_dim %1000, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1002 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1003 = stablehlo.divide %1001, %1002 : tensor<1x8x1xf32>
    %1004 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1005 = stablehlo.add %1003, %1004 : tensor<1x8x1xf32>
    %1006 = stablehlo.rsqrt %1005 : tensor<1x8x1xf32>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1008 = stablehlo.multiply %998, %1007 : tensor<1x8x1024xf32>
    %1009 = stablehlo.convert %1008 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1010 = stablehlo.convert %arg60 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1011 = stablehlo.broadcast_in_dim %1010, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1012 = stablehlo.broadcast_in_dim %1011, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1013 = stablehlo.multiply %1012, %1009 : tensor<1x8x1024xbf16>
    %1014 = stablehlo.convert %arg58 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1015 = stablehlo.dot_general %1013, %1014, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1016 = call @silu(%1015) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1017 = stablehlo.convert %arg59 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1018 = stablehlo.dot_general %1013, %1017, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1019 = stablehlo.multiply %1016, %1018 : tensor<1x8x3072xbf16>
    %1020 = stablehlo.convert %arg57 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1021 = stablehlo.dot_general %1019, %1020, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1022 = stablehlo.add %997, %1021 : tensor<1x8x1024xbf16>
    %1023 = stablehlo.convert %1022 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1024 = stablehlo.multiply %1023, %1023 : tensor<1x8x1024xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1025 = stablehlo.reduce(%1024 init: %cst_52) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1026 = stablehlo.broadcast_in_dim %1025, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1027 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1028 = stablehlo.divide %1026, %1027 : tensor<1x8x1xf32>
    %1029 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1030 = stablehlo.add %1028, %1029 : tensor<1x8x1xf32>
    %1031 = stablehlo.rsqrt %1030 : tensor<1x8x1xf32>
    %1032 = stablehlo.broadcast_in_dim %1031, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1033 = stablehlo.multiply %1023, %1032 : tensor<1x8x1024xf32>
    %1034 = stablehlo.convert %1033 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1035 = stablehlo.convert %arg67 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1036 = stablehlo.broadcast_in_dim %1035, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1037 = stablehlo.broadcast_in_dim %1036, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1038 = stablehlo.multiply %1037, %1034 : tensor<1x8x1024xbf16>
    %1039 = stablehlo.convert %arg76 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1040 = stablehlo.dot_general %1038, %1039, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1041 = stablehlo.convert %arg73 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1042 = stablehlo.dot_general %1038, %1041, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1043 = stablehlo.convert %arg77 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1044 = stablehlo.dot_general %1038, %1043, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1045 = stablehlo.reshape %1040 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1046 = stablehlo.convert %1045 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1047 = stablehlo.multiply %1046, %1046 : tensor<1x8x16x128xf32>
    %cst_53 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1048 = stablehlo.reduce(%1047 init: %cst_53) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1049 = stablehlo.broadcast_in_dim %1048, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1050 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1051 = stablehlo.divide %1049, %1050 : tensor<1x8x16x1xf32>
    %1052 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1053 = stablehlo.add %1051, %1052 : tensor<1x8x16x1xf32>
    %1054 = stablehlo.rsqrt %1053 : tensor<1x8x16x1xf32>
    %1055 = stablehlo.broadcast_in_dim %1054, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1056 = stablehlo.multiply %1046, %1055 : tensor<1x8x16x128xf32>
    %1057 = stablehlo.convert %1056 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1058 = stablehlo.convert %arg75 : (tensor<128xf32>) -> tensor<128xbf16>
    %1059 = stablehlo.broadcast_in_dim %1058, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1060 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1061 = stablehlo.multiply %1060, %1057 : tensor<1x8x16x128xbf16>
    %1062 = stablehlo.reshape %1042 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1063 = stablehlo.convert %1062 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1064 = stablehlo.multiply %1063, %1063 : tensor<1x8x8x128xf32>
    %cst_54 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1065 = stablehlo.reduce(%1064 init: %cst_54) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1066 = stablehlo.broadcast_in_dim %1065, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1067 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1068 = stablehlo.divide %1066, %1067 : tensor<1x8x8x1xf32>
    %1069 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1070 = stablehlo.add %1068, %1069 : tensor<1x8x8x1xf32>
    %1071 = stablehlo.rsqrt %1070 : tensor<1x8x8x1xf32>
    %1072 = stablehlo.broadcast_in_dim %1071, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1073 = stablehlo.multiply %1063, %1072 : tensor<1x8x8x128xf32>
    %1074 = stablehlo.convert %1073 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1075 = stablehlo.convert %arg72 : (tensor<128xf32>) -> tensor<128xbf16>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1078 = stablehlo.multiply %1077, %1074 : tensor<1x8x8x128xbf16>
    %1079 = stablehlo.reshape %1044 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1080 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1081 = stablehlo.compare  LT, %7, %1080,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1082 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1083 = stablehlo.add %7, %1082 : tensor<1x8xi32>
    %1084 = stablehlo.select %1081, %1083, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1085 = stablehlo.broadcast_in_dim %1084, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1086 = "stablehlo.gather"(%26, %1085) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1087 = stablehlo.slice %1086 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1088 = stablehlo.slice %1086 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1089 = stablehlo.broadcast_in_dim %1087, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1090 = stablehlo.convert %1089 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1091 = stablehlo.broadcast_in_dim %1088, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1092 = stablehlo.convert %1091 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1093 = stablehlo.slice %1061 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1094 = stablehlo.slice %1061 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1095 = stablehlo.broadcast_in_dim %1090, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1096 = stablehlo.multiply %1093, %1095 : tensor<1x8x16x64xbf16>
    %1097 = stablehlo.broadcast_in_dim %1092, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1098 = stablehlo.multiply %1094, %1097 : tensor<1x8x16x64xbf16>
    %1099 = stablehlo.subtract %1096, %1098 : tensor<1x8x16x64xbf16>
    %1100 = stablehlo.broadcast_in_dim %1090, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1101 = stablehlo.multiply %1094, %1100 : tensor<1x8x16x64xbf16>
    %1102 = stablehlo.broadcast_in_dim %1092, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1103 = stablehlo.multiply %1093, %1102 : tensor<1x8x16x64xbf16>
    %1104 = stablehlo.add %1101, %1103 : tensor<1x8x16x64xbf16>
    %1105 = stablehlo.concatenate %1099, %1104, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1106 = stablehlo.broadcast_in_dim %1087, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1107 = stablehlo.convert %1106 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1108 = stablehlo.broadcast_in_dim %1088, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1109 = stablehlo.convert %1108 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1110 = stablehlo.slice %1078 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1111 = stablehlo.slice %1078 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1112 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1113 = stablehlo.multiply %1110, %1112 : tensor<1x8x8x64xbf16>
    %1114 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1115 = stablehlo.multiply %1111, %1114 : tensor<1x8x8x64xbf16>
    %1116 = stablehlo.subtract %1113, %1115 : tensor<1x8x8x64xbf16>
    %1117 = stablehlo.broadcast_in_dim %1107, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1118 = stablehlo.multiply %1111, %1117 : tensor<1x8x8x64xbf16>
    %1119 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1120 = stablehlo.multiply %1110, %1119 : tensor<1x8x8x64xbf16>
    %1121 = stablehlo.add %1118, %1120 : tensor<1x8x8x64xbf16>
    %1122 = stablehlo.concatenate %1116, %1121, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1123 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1124 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1125 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1126 = stablehlo.and %1125, %1124 : tensor<1x1x8x8xi1>
    %1127 = stablehlo.convert %1126 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1128 = sdy.sharding_constraint %1105 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1129 = sdy.sharding_constraint %1122 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1130 = sdy.sharding_constraint %1079 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1131 = sdy.sharding_constraint %1127 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1132 = stablehlo.reshape %1128 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1133 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1134 = stablehlo.multiply %1132, %1133 : tensor<1x8x8x2x128xbf16>
    %1135 = stablehlo.dot_general %1129, %1134, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1136 = stablehlo.transpose %1135, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1137 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1138 = stablehlo.compare  NE, %1131, %1137,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1139 = stablehlo.convert %1138 : tensor<1x1x8x8xi1>
    %1140 = stablehlo.broadcast_in_dim %1139, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1141 = stablehlo.reshape %1140 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1142 = call @_where_83(%1141, %1136, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1143 = stablehlo.convert %1142 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_56 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1144 = stablehlo.reduce(%1143 init: %cst_56) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1145 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1146 = stablehlo.maximum %1145, %1144 : tensor<1x8x2x8xf32>
    %1147 = stablehlo.broadcast_in_dim %1146, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1148 = stablehlo.broadcast_in_dim %1147, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1149 = stablehlo.subtract %1143, %1148 : tensor<1x8x2x8x8xf32>
    %1150 = stablehlo.exponential %1149 : tensor<1x8x2x8x8xf32>
    %cst_57 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1151 = stablehlo.reduce(%1150 init: %cst_57) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1152 = stablehlo.broadcast_in_dim %1151, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1154 = stablehlo.divide %1150, %1153 : tensor<1x8x2x8x8xf32>
    %1155 = stablehlo.convert %1154 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1156 = stablehlo.dot_general %1130, %1155, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1157 = stablehlo.transpose %1156, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1158 = stablehlo.reshape %1157 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1159 = sdy.sharding_constraint %1158 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1160 = stablehlo.reshape %1159 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1161 = stablehlo.convert %arg74 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1162 = stablehlo.dot_general %1160, %1161, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1163 = stablehlo.add %1022, %1162 : tensor<1x8x1024xbf16>
    %1164 = stablehlo.convert %1163 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1165 = stablehlo.multiply %1164, %1164 : tensor<1x8x1024xf32>
    %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1166 = stablehlo.reduce(%1165 init: %cst_58) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1167 = stablehlo.broadcast_in_dim %1166, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1168 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1169 = stablehlo.divide %1167, %1168 : tensor<1x8x1xf32>
    %1170 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1171 = stablehlo.add %1169, %1170 : tensor<1x8x1xf32>
    %1172 = stablehlo.rsqrt %1171 : tensor<1x8x1xf32>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1174 = stablehlo.multiply %1164, %1173 : tensor<1x8x1024xf32>
    %1175 = stablehlo.convert %1174 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1176 = stablehlo.convert %arg71 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1177 = stablehlo.broadcast_in_dim %1176, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1178 = stablehlo.broadcast_in_dim %1177, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1179 = stablehlo.multiply %1178, %1175 : tensor<1x8x1024xbf16>
    %1180 = stablehlo.convert %arg69 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1181 = stablehlo.dot_general %1179, %1180, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1182 = call @silu(%1181) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1183 = stablehlo.convert %arg70 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1184 = stablehlo.dot_general %1179, %1183, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1185 = stablehlo.multiply %1182, %1184 : tensor<1x8x3072xbf16>
    %1186 = stablehlo.convert %arg68 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1187 = stablehlo.dot_general %1185, %1186, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1188 = stablehlo.add %1163, %1187 : tensor<1x8x1024xbf16>
    %1189 = stablehlo.convert %1188 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1190 = stablehlo.multiply %1189, %1189 : tensor<1x8x1024xf32>
    %cst_59 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1191 = stablehlo.reduce(%1190 init: %cst_59) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1193 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1194 = stablehlo.divide %1192, %1193 : tensor<1x8x1xf32>
    %1195 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1196 = stablehlo.add %1194, %1195 : tensor<1x8x1xf32>
    %1197 = stablehlo.rsqrt %1196 : tensor<1x8x1xf32>
    %1198 = stablehlo.broadcast_in_dim %1197, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1199 = stablehlo.multiply %1189, %1198 : tensor<1x8x1024xf32>
    %1200 = stablehlo.convert %1199 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1201 = stablehlo.convert %arg78 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1202 = stablehlo.broadcast_in_dim %1201, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1203 = stablehlo.broadcast_in_dim %1202, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1204 = stablehlo.multiply %1203, %1200 : tensor<1x8x1024xbf16>
    %1205 = stablehlo.convert %arg87 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1206 = stablehlo.dot_general %1204, %1205, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1207 = stablehlo.convert %arg84 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1208 = stablehlo.dot_general %1204, %1207, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1209 = stablehlo.convert %arg88 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1210 = stablehlo.dot_general %1204, %1209, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1211 = stablehlo.reshape %1206 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1212 = stablehlo.convert %1211 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1213 = stablehlo.multiply %1212, %1212 : tensor<1x8x16x128xf32>
    %cst_60 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1214 = stablehlo.reduce(%1213 init: %cst_60) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1215 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1216 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1217 = stablehlo.divide %1215, %1216 : tensor<1x8x16x1xf32>
    %1218 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1219 = stablehlo.add %1217, %1218 : tensor<1x8x16x1xf32>
    %1220 = stablehlo.rsqrt %1219 : tensor<1x8x16x1xf32>
    %1221 = stablehlo.broadcast_in_dim %1220, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1222 = stablehlo.multiply %1212, %1221 : tensor<1x8x16x128xf32>
    %1223 = stablehlo.convert %1222 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1224 = stablehlo.convert %arg86 : (tensor<128xf32>) -> tensor<128xbf16>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1226 = stablehlo.broadcast_in_dim %1225, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1227 = stablehlo.multiply %1226, %1223 : tensor<1x8x16x128xbf16>
    %1228 = stablehlo.reshape %1208 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1229 = stablehlo.convert %1228 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1230 = stablehlo.multiply %1229, %1229 : tensor<1x8x8x128xf32>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1231 = stablehlo.reduce(%1230 init: %cst_61) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1232 = stablehlo.broadcast_in_dim %1231, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1233 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1234 = stablehlo.divide %1232, %1233 : tensor<1x8x8x1xf32>
    %1235 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1236 = stablehlo.add %1234, %1235 : tensor<1x8x8x1xf32>
    %1237 = stablehlo.rsqrt %1236 : tensor<1x8x8x1xf32>
    %1238 = stablehlo.broadcast_in_dim %1237, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1239 = stablehlo.multiply %1229, %1238 : tensor<1x8x8x128xf32>
    %1240 = stablehlo.convert %1239 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1241 = stablehlo.convert %arg83 : (tensor<128xf32>) -> tensor<128xbf16>
    %1242 = stablehlo.broadcast_in_dim %1241, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1243 = stablehlo.broadcast_in_dim %1242, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1244 = stablehlo.multiply %1243, %1240 : tensor<1x8x8x128xbf16>
    %1245 = stablehlo.reshape %1210 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1246 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1247 = stablehlo.compare  LT, %7, %1246,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1248 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1249 = stablehlo.add %7, %1248 : tensor<1x8xi32>
    %1250 = stablehlo.select %1247, %1249, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1251 = stablehlo.broadcast_in_dim %1250, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1252 = "stablehlo.gather"(%26, %1251) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1253 = stablehlo.slice %1252 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1254 = stablehlo.slice %1252 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1255 = stablehlo.broadcast_in_dim %1253, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1256 = stablehlo.convert %1255 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1257 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1258 = stablehlo.convert %1257 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1259 = stablehlo.slice %1227 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1260 = stablehlo.slice %1227 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1261 = stablehlo.broadcast_in_dim %1256, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1262 = stablehlo.multiply %1259, %1261 : tensor<1x8x16x64xbf16>
    %1263 = stablehlo.broadcast_in_dim %1258, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1264 = stablehlo.multiply %1260, %1263 : tensor<1x8x16x64xbf16>
    %1265 = stablehlo.subtract %1262, %1264 : tensor<1x8x16x64xbf16>
    %1266 = stablehlo.broadcast_in_dim %1256, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1267 = stablehlo.multiply %1260, %1266 : tensor<1x8x16x64xbf16>
    %1268 = stablehlo.broadcast_in_dim %1258, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1269 = stablehlo.multiply %1259, %1268 : tensor<1x8x16x64xbf16>
    %1270 = stablehlo.add %1267, %1269 : tensor<1x8x16x64xbf16>
    %1271 = stablehlo.concatenate %1265, %1270, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1272 = stablehlo.broadcast_in_dim %1253, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1273 = stablehlo.convert %1272 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1274 = stablehlo.broadcast_in_dim %1254, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1275 = stablehlo.convert %1274 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1276 = stablehlo.slice %1244 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1277 = stablehlo.slice %1244 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1278 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1279 = stablehlo.multiply %1276, %1278 : tensor<1x8x8x64xbf16>
    %1280 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1281 = stablehlo.multiply %1277, %1280 : tensor<1x8x8x64xbf16>
    %1282 = stablehlo.subtract %1279, %1281 : tensor<1x8x8x64xbf16>
    %1283 = stablehlo.broadcast_in_dim %1273, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1284 = stablehlo.multiply %1277, %1283 : tensor<1x8x8x64xbf16>
    %1285 = stablehlo.broadcast_in_dim %1275, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1286 = stablehlo.multiply %1276, %1285 : tensor<1x8x8x64xbf16>
    %1287 = stablehlo.add %1284, %1286 : tensor<1x8x8x64xbf16>
    %1288 = stablehlo.concatenate %1282, %1287, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1289 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1290 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1291 = stablehlo.broadcast_in_dim %1289, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1292 = stablehlo.and %1291, %1290 : tensor<1x1x8x8xi1>
    %1293 = stablehlo.convert %1292 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1294 = sdy.sharding_constraint %1271 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1295 = sdy.sharding_constraint %1288 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1296 = sdy.sharding_constraint %1245 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1297 = sdy.sharding_constraint %1293 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1298 = stablehlo.reshape %1294 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1299 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1300 = stablehlo.multiply %1298, %1299 : tensor<1x8x8x2x128xbf16>
    %1301 = stablehlo.dot_general %1295, %1300, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1302 = stablehlo.transpose %1301, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_62 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1303 = stablehlo.broadcast_in_dim %cst_62, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1304 = stablehlo.compare  NE, %1297, %1303,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1305 = stablehlo.convert %1304 : tensor<1x1x8x8xi1>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1307 = stablehlo.reshape %1306 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1308 = call @_where_83(%1307, %1302, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1309 = stablehlo.convert %1308 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_63 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1310 = stablehlo.reduce(%1309 init: %cst_63) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1311 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1312 = stablehlo.maximum %1311, %1310 : tensor<1x8x2x8xf32>
    %1313 = stablehlo.broadcast_in_dim %1312, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1314 = stablehlo.broadcast_in_dim %1313, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1315 = stablehlo.subtract %1309, %1314 : tensor<1x8x2x8x8xf32>
    %1316 = stablehlo.exponential %1315 : tensor<1x8x2x8x8xf32>
    %cst_64 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1317 = stablehlo.reduce(%1316 init: %cst_64) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1318 = stablehlo.broadcast_in_dim %1317, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1319 = stablehlo.broadcast_in_dim %1318, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1320 = stablehlo.divide %1316, %1319 : tensor<1x8x2x8x8xf32>
    %1321 = stablehlo.convert %1320 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1322 = stablehlo.dot_general %1296, %1321, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1323 = stablehlo.transpose %1322, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1324 = stablehlo.reshape %1323 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1325 = sdy.sharding_constraint %1324 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1326 = stablehlo.reshape %1325 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1327 = stablehlo.convert %arg85 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1328 = stablehlo.dot_general %1326, %1327, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1329 = stablehlo.add %1188, %1328 : tensor<1x8x1024xbf16>
    %1330 = stablehlo.convert %1329 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1331 = stablehlo.multiply %1330, %1330 : tensor<1x8x1024xf32>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1332 = stablehlo.reduce(%1331 init: %cst_65) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1333 = stablehlo.broadcast_in_dim %1332, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1334 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1335 = stablehlo.divide %1333, %1334 : tensor<1x8x1xf32>
    %1336 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1337 = stablehlo.add %1335, %1336 : tensor<1x8x1xf32>
    %1338 = stablehlo.rsqrt %1337 : tensor<1x8x1xf32>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1340 = stablehlo.multiply %1330, %1339 : tensor<1x8x1024xf32>
    %1341 = stablehlo.convert %1340 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1342 = stablehlo.convert %arg82 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1343 = stablehlo.broadcast_in_dim %1342, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1345 = stablehlo.multiply %1344, %1341 : tensor<1x8x1024xbf16>
    %1346 = stablehlo.convert %arg80 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1347 = stablehlo.dot_general %1345, %1346, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1348 = call @silu(%1347) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1349 = stablehlo.convert %arg81 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1350 = stablehlo.dot_general %1345, %1349, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1351 = stablehlo.multiply %1348, %1350 : tensor<1x8x3072xbf16>
    %1352 = stablehlo.convert %arg79 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1353 = stablehlo.dot_general %1351, %1352, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1354 = stablehlo.add %1329, %1353 : tensor<1x8x1024xbf16>
    %1355 = stablehlo.convert %1354 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1356 = stablehlo.multiply %1355, %1355 : tensor<1x8x1024xf32>
    %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1357 = stablehlo.reduce(%1356 init: %cst_66) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1358 = stablehlo.broadcast_in_dim %1357, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1359 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1360 = stablehlo.divide %1358, %1359 : tensor<1x8x1xf32>
    %1361 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1362 = stablehlo.add %1360, %1361 : tensor<1x8x1xf32>
    %1363 = stablehlo.rsqrt %1362 : tensor<1x8x1xf32>
    %1364 = stablehlo.broadcast_in_dim %1363, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1365 = stablehlo.multiply %1355, %1364 : tensor<1x8x1024xf32>
    %1366 = stablehlo.convert %1365 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1367 = stablehlo.convert %arg89 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1368 = stablehlo.broadcast_in_dim %1367, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1369 = stablehlo.broadcast_in_dim %1368, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1370 = stablehlo.multiply %1369, %1366 : tensor<1x8x1024xbf16>
    %1371 = stablehlo.convert %arg98 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1372 = stablehlo.dot_general %1370, %1371, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1373 = stablehlo.convert %arg95 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1374 = stablehlo.dot_general %1370, %1373, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1375 = stablehlo.convert %arg99 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1376 = stablehlo.dot_general %1370, %1375, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1377 = stablehlo.reshape %1372 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1378 = stablehlo.convert %1377 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1379 = stablehlo.multiply %1378, %1378 : tensor<1x8x16x128xf32>
    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1380 = stablehlo.reduce(%1379 init: %cst_67) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1381 = stablehlo.broadcast_in_dim %1380, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1382 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1383 = stablehlo.divide %1381, %1382 : tensor<1x8x16x1xf32>
    %1384 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1385 = stablehlo.add %1383, %1384 : tensor<1x8x16x1xf32>
    %1386 = stablehlo.rsqrt %1385 : tensor<1x8x16x1xf32>
    %1387 = stablehlo.broadcast_in_dim %1386, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1388 = stablehlo.multiply %1378, %1387 : tensor<1x8x16x128xf32>
    %1389 = stablehlo.convert %1388 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1390 = stablehlo.convert %arg97 : (tensor<128xf32>) -> tensor<128xbf16>
    %1391 = stablehlo.broadcast_in_dim %1390, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1393 = stablehlo.multiply %1392, %1389 : tensor<1x8x16x128xbf16>
    %1394 = stablehlo.reshape %1374 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1395 = stablehlo.convert %1394 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1396 = stablehlo.multiply %1395, %1395 : tensor<1x8x8x128xf32>
    %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1397 = stablehlo.reduce(%1396 init: %cst_68) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1398 = stablehlo.broadcast_in_dim %1397, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1399 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1400 = stablehlo.divide %1398, %1399 : tensor<1x8x8x1xf32>
    %1401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1402 = stablehlo.add %1400, %1401 : tensor<1x8x8x1xf32>
    %1403 = stablehlo.rsqrt %1402 : tensor<1x8x8x1xf32>
    %1404 = stablehlo.broadcast_in_dim %1403, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1405 = stablehlo.multiply %1395, %1404 : tensor<1x8x8x128xf32>
    %1406 = stablehlo.convert %1405 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1407 = stablehlo.convert %arg94 : (tensor<128xf32>) -> tensor<128xbf16>
    %1408 = stablehlo.broadcast_in_dim %1407, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1409 = stablehlo.broadcast_in_dim %1408, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1410 = stablehlo.multiply %1409, %1406 : tensor<1x8x8x128xbf16>
    %1411 = stablehlo.reshape %1376 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1412 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1413 = stablehlo.compare  LT, %7, %1412,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1414 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1415 = stablehlo.add %7, %1414 : tensor<1x8xi32>
    %1416 = stablehlo.select %1413, %1415, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1417 = stablehlo.broadcast_in_dim %1416, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1418 = "stablehlo.gather"(%26, %1417) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1419 = stablehlo.slice %1418 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1420 = stablehlo.slice %1418 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1421 = stablehlo.broadcast_in_dim %1419, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1422 = stablehlo.convert %1421 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1423 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1424 = stablehlo.convert %1423 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1425 = stablehlo.slice %1393 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1426 = stablehlo.slice %1393 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1427 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1428 = stablehlo.multiply %1425, %1427 : tensor<1x8x16x64xbf16>
    %1429 = stablehlo.broadcast_in_dim %1424, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1430 = stablehlo.multiply %1426, %1429 : tensor<1x8x16x64xbf16>
    %1431 = stablehlo.subtract %1428, %1430 : tensor<1x8x16x64xbf16>
    %1432 = stablehlo.broadcast_in_dim %1422, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1433 = stablehlo.multiply %1426, %1432 : tensor<1x8x16x64xbf16>
    %1434 = stablehlo.broadcast_in_dim %1424, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1435 = stablehlo.multiply %1425, %1434 : tensor<1x8x16x64xbf16>
    %1436 = stablehlo.add %1433, %1435 : tensor<1x8x16x64xbf16>
    %1437 = stablehlo.concatenate %1431, %1436, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1438 = stablehlo.broadcast_in_dim %1419, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1439 = stablehlo.convert %1438 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1440 = stablehlo.broadcast_in_dim %1420, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1441 = stablehlo.convert %1440 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1442 = stablehlo.slice %1410 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1443 = stablehlo.slice %1410 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1444 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1445 = stablehlo.multiply %1442, %1444 : tensor<1x8x8x64xbf16>
    %1446 = stablehlo.broadcast_in_dim %1441, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1447 = stablehlo.multiply %1443, %1446 : tensor<1x8x8x64xbf16>
    %1448 = stablehlo.subtract %1445, %1447 : tensor<1x8x8x64xbf16>
    %1449 = stablehlo.broadcast_in_dim %1439, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1450 = stablehlo.multiply %1443, %1449 : tensor<1x8x8x64xbf16>
    %1451 = stablehlo.broadcast_in_dim %1441, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1452 = stablehlo.multiply %1442, %1451 : tensor<1x8x8x64xbf16>
    %1453 = stablehlo.add %1450, %1452 : tensor<1x8x8x64xbf16>
    %1454 = stablehlo.concatenate %1448, %1453, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1455 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1456 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1457 = stablehlo.broadcast_in_dim %1455, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1458 = stablehlo.and %1457, %1456 : tensor<1x1x8x8xi1>
    %1459 = stablehlo.convert %1458 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1460 = sdy.sharding_constraint %1437 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1461 = sdy.sharding_constraint %1454 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1462 = sdy.sharding_constraint %1411 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1463 = sdy.sharding_constraint %1459 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1464 = stablehlo.reshape %1460 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1465 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1466 = stablehlo.multiply %1464, %1465 : tensor<1x8x8x2x128xbf16>
    %1467 = stablehlo.dot_general %1461, %1466, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1468 = stablehlo.transpose %1467, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1469 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1470 = stablehlo.compare  NE, %1463, %1469,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1471 = stablehlo.convert %1470 : tensor<1x1x8x8xi1>
    %1472 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1473 = stablehlo.reshape %1472 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1474 = call @_where_83(%1473, %1468, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1475 = stablehlo.convert %1474 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_70 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1476 = stablehlo.reduce(%1475 init: %cst_70) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1477 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1478 = stablehlo.maximum %1477, %1476 : tensor<1x8x2x8xf32>
    %1479 = stablehlo.broadcast_in_dim %1478, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1480 = stablehlo.broadcast_in_dim %1479, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1481 = stablehlo.subtract %1475, %1480 : tensor<1x8x2x8x8xf32>
    %1482 = stablehlo.exponential %1481 : tensor<1x8x2x8x8xf32>
    %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1483 = stablehlo.reduce(%1482 init: %cst_71) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1484 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1486 = stablehlo.divide %1482, %1485 : tensor<1x8x2x8x8xf32>
    %1487 = stablehlo.convert %1486 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1488 = stablehlo.dot_general %1462, %1487, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1489 = stablehlo.transpose %1488, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1490 = stablehlo.reshape %1489 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1491 = sdy.sharding_constraint %1490 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1492 = stablehlo.reshape %1491 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1493 = stablehlo.convert %arg96 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1494 = stablehlo.dot_general %1492, %1493, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1495 = stablehlo.add %1354, %1494 : tensor<1x8x1024xbf16>
    %1496 = stablehlo.convert %1495 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1497 = stablehlo.multiply %1496, %1496 : tensor<1x8x1024xf32>
    %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1498 = stablehlo.reduce(%1497 init: %cst_72) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1499 = stablehlo.broadcast_in_dim %1498, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1500 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1501 = stablehlo.divide %1499, %1500 : tensor<1x8x1xf32>
    %1502 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1503 = stablehlo.add %1501, %1502 : tensor<1x8x1xf32>
    %1504 = stablehlo.rsqrt %1503 : tensor<1x8x1xf32>
    %1505 = stablehlo.broadcast_in_dim %1504, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1506 = stablehlo.multiply %1496, %1505 : tensor<1x8x1024xf32>
    %1507 = stablehlo.convert %1506 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1508 = stablehlo.convert %arg93 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1509 = stablehlo.broadcast_in_dim %1508, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1510 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1511 = stablehlo.multiply %1510, %1507 : tensor<1x8x1024xbf16>
    %1512 = stablehlo.convert %arg91 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1513 = stablehlo.dot_general %1511, %1512, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1514 = call @silu(%1513) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1515 = stablehlo.convert %arg92 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1516 = stablehlo.dot_general %1511, %1515, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1517 = stablehlo.multiply %1514, %1516 : tensor<1x8x3072xbf16>
    %1518 = stablehlo.convert %arg90 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1519 = stablehlo.dot_general %1517, %1518, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1520 = stablehlo.add %1495, %1519 : tensor<1x8x1024xbf16>
    %1521 = stablehlo.convert %1520 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1522 = stablehlo.multiply %1521, %1521 : tensor<1x8x1024xf32>
    %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1523 = stablehlo.reduce(%1522 init: %cst_73) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1524 = stablehlo.broadcast_in_dim %1523, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1525 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1526 = stablehlo.divide %1524, %1525 : tensor<1x8x1xf32>
    %1527 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1528 = stablehlo.add %1526, %1527 : tensor<1x8x1xf32>
    %1529 = stablehlo.rsqrt %1528 : tensor<1x8x1xf32>
    %1530 = stablehlo.broadcast_in_dim %1529, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1531 = stablehlo.multiply %1521, %1530 : tensor<1x8x1024xf32>
    %1532 = stablehlo.convert %1531 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1533 = stablehlo.convert %arg100 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1536 = stablehlo.multiply %1535, %1532 : tensor<1x8x1024xbf16>
    %1537 = stablehlo.convert %arg109 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1538 = stablehlo.dot_general %1536, %1537, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1539 = stablehlo.convert %arg106 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1540 = stablehlo.dot_general %1536, %1539, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1541 = stablehlo.convert %arg110 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1542 = stablehlo.dot_general %1536, %1541, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1543 = stablehlo.reshape %1538 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1544 = stablehlo.convert %1543 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1545 = stablehlo.multiply %1544, %1544 : tensor<1x8x16x128xf32>
    %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1546 = stablehlo.reduce(%1545 init: %cst_74) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1547 = stablehlo.broadcast_in_dim %1546, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1548 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1549 = stablehlo.divide %1547, %1548 : tensor<1x8x16x1xf32>
    %1550 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1551 = stablehlo.add %1549, %1550 : tensor<1x8x16x1xf32>
    %1552 = stablehlo.rsqrt %1551 : tensor<1x8x16x1xf32>
    %1553 = stablehlo.broadcast_in_dim %1552, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1554 = stablehlo.multiply %1544, %1553 : tensor<1x8x16x128xf32>
    %1555 = stablehlo.convert %1554 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1556 = stablehlo.convert %arg108 : (tensor<128xf32>) -> tensor<128xbf16>
    %1557 = stablehlo.broadcast_in_dim %1556, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1558 = stablehlo.broadcast_in_dim %1557, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1559 = stablehlo.multiply %1558, %1555 : tensor<1x8x16x128xbf16>
    %1560 = stablehlo.reshape %1540 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1561 = stablehlo.convert %1560 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1562 = stablehlo.multiply %1561, %1561 : tensor<1x8x8x128xf32>
    %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1563 = stablehlo.reduce(%1562 init: %cst_75) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1565 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1566 = stablehlo.divide %1564, %1565 : tensor<1x8x8x1xf32>
    %1567 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1568 = stablehlo.add %1566, %1567 : tensor<1x8x8x1xf32>
    %1569 = stablehlo.rsqrt %1568 : tensor<1x8x8x1xf32>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1571 = stablehlo.multiply %1561, %1570 : tensor<1x8x8x128xf32>
    %1572 = stablehlo.convert %1571 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1573 = stablehlo.convert %arg105 : (tensor<128xf32>) -> tensor<128xbf16>
    %1574 = stablehlo.broadcast_in_dim %1573, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1575 = stablehlo.broadcast_in_dim %1574, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1576 = stablehlo.multiply %1575, %1572 : tensor<1x8x8x128xbf16>
    %1577 = stablehlo.reshape %1542 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1578 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1579 = stablehlo.compare  LT, %7, %1578,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1580 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1581 = stablehlo.add %7, %1580 : tensor<1x8xi32>
    %1582 = stablehlo.select %1579, %1581, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1583 = stablehlo.broadcast_in_dim %1582, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1584 = "stablehlo.gather"(%26, %1583) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1585 = stablehlo.slice %1584 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1586 = stablehlo.slice %1584 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1587 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1588 = stablehlo.convert %1587 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1589 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1590 = stablehlo.convert %1589 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1591 = stablehlo.slice %1559 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1592 = stablehlo.slice %1559 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1593 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1594 = stablehlo.multiply %1591, %1593 : tensor<1x8x16x64xbf16>
    %1595 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1596 = stablehlo.multiply %1592, %1595 : tensor<1x8x16x64xbf16>
    %1597 = stablehlo.subtract %1594, %1596 : tensor<1x8x16x64xbf16>
    %1598 = stablehlo.broadcast_in_dim %1588, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1599 = stablehlo.multiply %1592, %1598 : tensor<1x8x16x64xbf16>
    %1600 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1601 = stablehlo.multiply %1591, %1600 : tensor<1x8x16x64xbf16>
    %1602 = stablehlo.add %1599, %1601 : tensor<1x8x16x64xbf16>
    %1603 = stablehlo.concatenate %1597, %1602, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1604 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1605 = stablehlo.convert %1604 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1606 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1607 = stablehlo.convert %1606 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1608 = stablehlo.slice %1576 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1609 = stablehlo.slice %1576 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1610 = stablehlo.broadcast_in_dim %1605, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1611 = stablehlo.multiply %1608, %1610 : tensor<1x8x8x64xbf16>
    %1612 = stablehlo.broadcast_in_dim %1607, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1613 = stablehlo.multiply %1609, %1612 : tensor<1x8x8x64xbf16>
    %1614 = stablehlo.subtract %1611, %1613 : tensor<1x8x8x64xbf16>
    %1615 = stablehlo.broadcast_in_dim %1605, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1616 = stablehlo.multiply %1609, %1615 : tensor<1x8x8x64xbf16>
    %1617 = stablehlo.broadcast_in_dim %1607, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1618 = stablehlo.multiply %1608, %1617 : tensor<1x8x8x64xbf16>
    %1619 = stablehlo.add %1616, %1618 : tensor<1x8x8x64xbf16>
    %1620 = stablehlo.concatenate %1614, %1619, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1621 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1622 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1623 = stablehlo.broadcast_in_dim %1621, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1624 = stablehlo.and %1623, %1622 : tensor<1x1x8x8xi1>
    %1625 = stablehlo.convert %1624 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1626 = sdy.sharding_constraint %1603 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1627 = sdy.sharding_constraint %1620 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1628 = sdy.sharding_constraint %1577 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1629 = sdy.sharding_constraint %1625 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1630 = stablehlo.reshape %1626 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1631 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1632 = stablehlo.multiply %1630, %1631 : tensor<1x8x8x2x128xbf16>
    %1633 = stablehlo.dot_general %1627, %1632, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1634 = stablehlo.transpose %1633, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1635 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1636 = stablehlo.compare  NE, %1629, %1635,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1637 = stablehlo.convert %1636 : tensor<1x1x8x8xi1>
    %1638 = stablehlo.broadcast_in_dim %1637, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1639 = stablehlo.reshape %1638 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1640 = call @_where_83(%1639, %1634, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1641 = stablehlo.convert %1640 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_77 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1642 = stablehlo.reduce(%1641 init: %cst_77) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1643 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1644 = stablehlo.maximum %1643, %1642 : tensor<1x8x2x8xf32>
    %1645 = stablehlo.broadcast_in_dim %1644, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1646 = stablehlo.broadcast_in_dim %1645, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1647 = stablehlo.subtract %1641, %1646 : tensor<1x8x2x8x8xf32>
    %1648 = stablehlo.exponential %1647 : tensor<1x8x2x8x8xf32>
    %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1649 = stablehlo.reduce(%1648 init: %cst_78) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1650 = stablehlo.broadcast_in_dim %1649, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1651 = stablehlo.broadcast_in_dim %1650, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1652 = stablehlo.divide %1648, %1651 : tensor<1x8x2x8x8xf32>
    %1653 = stablehlo.convert %1652 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1654 = stablehlo.dot_general %1628, %1653, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1655 = stablehlo.transpose %1654, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1656 = stablehlo.reshape %1655 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1657 = sdy.sharding_constraint %1656 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1658 = stablehlo.reshape %1657 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1659 = stablehlo.convert %arg107 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1660 = stablehlo.dot_general %1658, %1659, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1661 = stablehlo.add %1520, %1660 : tensor<1x8x1024xbf16>
    %1662 = stablehlo.convert %1661 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1663 = stablehlo.multiply %1662, %1662 : tensor<1x8x1024xf32>
    %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1664 = stablehlo.reduce(%1663 init: %cst_79) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1665 = stablehlo.broadcast_in_dim %1664, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1666 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1667 = stablehlo.divide %1665, %1666 : tensor<1x8x1xf32>
    %1668 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1669 = stablehlo.add %1667, %1668 : tensor<1x8x1xf32>
    %1670 = stablehlo.rsqrt %1669 : tensor<1x8x1xf32>
    %1671 = stablehlo.broadcast_in_dim %1670, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1672 = stablehlo.multiply %1662, %1671 : tensor<1x8x1024xf32>
    %1673 = stablehlo.convert %1672 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1674 = stablehlo.convert %arg104 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1675 = stablehlo.broadcast_in_dim %1674, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1676 = stablehlo.broadcast_in_dim %1675, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1677 = stablehlo.multiply %1676, %1673 : tensor<1x8x1024xbf16>
    %1678 = stablehlo.convert %arg102 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1679 = stablehlo.dot_general %1677, %1678, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1680 = call @silu(%1679) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1681 = stablehlo.convert %arg103 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1682 = stablehlo.dot_general %1677, %1681, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1683 = stablehlo.multiply %1680, %1682 : tensor<1x8x3072xbf16>
    %1684 = stablehlo.convert %arg101 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1685 = stablehlo.dot_general %1683, %1684, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1686 = stablehlo.add %1661, %1685 : tensor<1x8x1024xbf16>
    %1687 = stablehlo.convert %1686 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1688 = stablehlo.multiply %1687, %1687 : tensor<1x8x1024xf32>
    %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1689 = stablehlo.reduce(%1688 init: %cst_80) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1690 = stablehlo.broadcast_in_dim %1689, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1691 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1692 = stablehlo.divide %1690, %1691 : tensor<1x8x1xf32>
    %1693 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1694 = stablehlo.add %1692, %1693 : tensor<1x8x1xf32>
    %1695 = stablehlo.rsqrt %1694 : tensor<1x8x1xf32>
    %1696 = stablehlo.broadcast_in_dim %1695, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1697 = stablehlo.multiply %1687, %1696 : tensor<1x8x1024xf32>
    %1698 = stablehlo.convert %1697 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1699 = stablehlo.convert %arg111 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1701 = stablehlo.broadcast_in_dim %1700, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1702 = stablehlo.multiply %1701, %1698 : tensor<1x8x1024xbf16>
    %1703 = stablehlo.convert %arg120 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1704 = stablehlo.dot_general %1702, %1703, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1705 = stablehlo.convert %arg117 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1706 = stablehlo.dot_general %1702, %1705, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1707 = stablehlo.convert %arg121 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1708 = stablehlo.dot_general %1702, %1707, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1709 = stablehlo.reshape %1704 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1710 = stablehlo.convert %1709 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1711 = stablehlo.multiply %1710, %1710 : tensor<1x8x16x128xf32>
    %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1712 = stablehlo.reduce(%1711 init: %cst_81) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1713 = stablehlo.broadcast_in_dim %1712, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1714 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1715 = stablehlo.divide %1713, %1714 : tensor<1x8x16x1xf32>
    %1716 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1717 = stablehlo.add %1715, %1716 : tensor<1x8x16x1xf32>
    %1718 = stablehlo.rsqrt %1717 : tensor<1x8x16x1xf32>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1720 = stablehlo.multiply %1710, %1719 : tensor<1x8x16x128xf32>
    %1721 = stablehlo.convert %1720 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1722 = stablehlo.convert %arg119 : (tensor<128xf32>) -> tensor<128xbf16>
    %1723 = stablehlo.broadcast_in_dim %1722, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1724 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1725 = stablehlo.multiply %1724, %1721 : tensor<1x8x16x128xbf16>
    %1726 = stablehlo.reshape %1706 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1727 = stablehlo.convert %1726 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1728 = stablehlo.multiply %1727, %1727 : tensor<1x8x8x128xf32>
    %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1729 = stablehlo.reduce(%1728 init: %cst_82) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1730 = stablehlo.broadcast_in_dim %1729, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1731 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1732 = stablehlo.divide %1730, %1731 : tensor<1x8x8x1xf32>
    %1733 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1734 = stablehlo.add %1732, %1733 : tensor<1x8x8x1xf32>
    %1735 = stablehlo.rsqrt %1734 : tensor<1x8x8x1xf32>
    %1736 = stablehlo.broadcast_in_dim %1735, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1737 = stablehlo.multiply %1727, %1736 : tensor<1x8x8x128xf32>
    %1738 = stablehlo.convert %1737 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1739 = stablehlo.convert %arg116 : (tensor<128xf32>) -> tensor<128xbf16>
    %1740 = stablehlo.broadcast_in_dim %1739, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1741 = stablehlo.broadcast_in_dim %1740, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1742 = stablehlo.multiply %1741, %1738 : tensor<1x8x8x128xbf16>
    %1743 = stablehlo.reshape %1708 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1744 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1745 = stablehlo.compare  LT, %7, %1744,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1746 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1747 = stablehlo.add %7, %1746 : tensor<1x8xi32>
    %1748 = stablehlo.select %1745, %1747, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1749 = stablehlo.broadcast_in_dim %1748, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1750 = "stablehlo.gather"(%26, %1749) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1751 = stablehlo.slice %1750 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1752 = stablehlo.slice %1750 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1753 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1754 = stablehlo.convert %1753 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1755 = stablehlo.broadcast_in_dim %1752, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1756 = stablehlo.convert %1755 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1757 = stablehlo.slice %1725 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1758 = stablehlo.slice %1725 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1759 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1760 = stablehlo.multiply %1757, %1759 : tensor<1x8x16x64xbf16>
    %1761 = stablehlo.broadcast_in_dim %1756, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1762 = stablehlo.multiply %1758, %1761 : tensor<1x8x16x64xbf16>
    %1763 = stablehlo.subtract %1760, %1762 : tensor<1x8x16x64xbf16>
    %1764 = stablehlo.broadcast_in_dim %1754, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1765 = stablehlo.multiply %1758, %1764 : tensor<1x8x16x64xbf16>
    %1766 = stablehlo.broadcast_in_dim %1756, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1767 = stablehlo.multiply %1757, %1766 : tensor<1x8x16x64xbf16>
    %1768 = stablehlo.add %1765, %1767 : tensor<1x8x16x64xbf16>
    %1769 = stablehlo.concatenate %1763, %1768, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1770 = stablehlo.broadcast_in_dim %1751, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1771 = stablehlo.convert %1770 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1772 = stablehlo.broadcast_in_dim %1752, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1773 = stablehlo.convert %1772 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1774 = stablehlo.slice %1742 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1775 = stablehlo.slice %1742 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1776 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1777 = stablehlo.multiply %1774, %1776 : tensor<1x8x8x64xbf16>
    %1778 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1779 = stablehlo.multiply %1775, %1778 : tensor<1x8x8x64xbf16>
    %1780 = stablehlo.subtract %1777, %1779 : tensor<1x8x8x64xbf16>
    %1781 = stablehlo.broadcast_in_dim %1771, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1782 = stablehlo.multiply %1775, %1781 : tensor<1x8x8x64xbf16>
    %1783 = stablehlo.broadcast_in_dim %1773, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1784 = stablehlo.multiply %1774, %1783 : tensor<1x8x8x64xbf16>
    %1785 = stablehlo.add %1782, %1784 : tensor<1x8x8x64xbf16>
    %1786 = stablehlo.concatenate %1780, %1785, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1787 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1788 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1789 = stablehlo.broadcast_in_dim %1787, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1790 = stablehlo.and %1789, %1788 : tensor<1x1x8x8xi1>
    %1791 = stablehlo.convert %1790 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1792 = sdy.sharding_constraint %1769 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1793 = sdy.sharding_constraint %1786 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1794 = sdy.sharding_constraint %1743 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1795 = sdy.sharding_constraint %1791 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1796 = stablehlo.reshape %1792 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1797 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1798 = stablehlo.multiply %1796, %1797 : tensor<1x8x8x2x128xbf16>
    %1799 = stablehlo.dot_general %1793, %1798, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1800 = stablehlo.transpose %1799, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1801 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1802 = stablehlo.compare  NE, %1795, %1801,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1803 = stablehlo.convert %1802 : tensor<1x1x8x8xi1>
    %1804 = stablehlo.broadcast_in_dim %1803, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1805 = stablehlo.reshape %1804 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1806 = call @_where_83(%1805, %1800, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1807 = stablehlo.convert %1806 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_84 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1808 = stablehlo.reduce(%1807 init: %cst_84) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1809 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1810 = stablehlo.maximum %1809, %1808 : tensor<1x8x2x8xf32>
    %1811 = stablehlo.broadcast_in_dim %1810, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1812 = stablehlo.broadcast_in_dim %1811, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1813 = stablehlo.subtract %1807, %1812 : tensor<1x8x2x8x8xf32>
    %1814 = stablehlo.exponential %1813 : tensor<1x8x2x8x8xf32>
    %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1815 = stablehlo.reduce(%1814 init: %cst_85) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1816 = stablehlo.broadcast_in_dim %1815, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1817 = stablehlo.broadcast_in_dim %1816, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1818 = stablehlo.divide %1814, %1817 : tensor<1x8x2x8x8xf32>
    %1819 = stablehlo.convert %1818 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1820 = stablehlo.dot_general %1794, %1819, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1821 = stablehlo.transpose %1820, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1822 = stablehlo.reshape %1821 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1823 = sdy.sharding_constraint %1822 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1824 = stablehlo.reshape %1823 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1825 = stablehlo.convert %arg118 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1826 = stablehlo.dot_general %1824, %1825, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1827 = stablehlo.add %1686, %1826 : tensor<1x8x1024xbf16>
    %1828 = stablehlo.convert %1827 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1829 = stablehlo.multiply %1828, %1828 : tensor<1x8x1024xf32>
    %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1830 = stablehlo.reduce(%1829 init: %cst_86) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1831 = stablehlo.broadcast_in_dim %1830, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1832 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1833 = stablehlo.divide %1831, %1832 : tensor<1x8x1xf32>
    %1834 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1835 = stablehlo.add %1833, %1834 : tensor<1x8x1xf32>
    %1836 = stablehlo.rsqrt %1835 : tensor<1x8x1xf32>
    %1837 = stablehlo.broadcast_in_dim %1836, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1838 = stablehlo.multiply %1828, %1837 : tensor<1x8x1024xf32>
    %1839 = stablehlo.convert %1838 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1840 = stablehlo.convert %arg115 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1841 = stablehlo.broadcast_in_dim %1840, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1842 = stablehlo.broadcast_in_dim %1841, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1843 = stablehlo.multiply %1842, %1839 : tensor<1x8x1024xbf16>
    %1844 = stablehlo.convert %arg113 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1845 = stablehlo.dot_general %1843, %1844, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1846 = call @silu(%1845) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1847 = stablehlo.convert %arg114 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %1848 = stablehlo.dot_general %1843, %1847, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %1849 = stablehlo.multiply %1846, %1848 : tensor<1x8x3072xbf16>
    %1850 = stablehlo.convert %arg112 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %1851 = stablehlo.dot_general %1849, %1850, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1852 = stablehlo.add %1827, %1851 : tensor<1x8x1024xbf16>
    %1853 = stablehlo.convert %1852 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1854 = stablehlo.multiply %1853, %1853 : tensor<1x8x1024xf32>
    %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1855 = stablehlo.reduce(%1854 init: %cst_87) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1856 = stablehlo.broadcast_in_dim %1855, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1857 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1858 = stablehlo.divide %1856, %1857 : tensor<1x8x1xf32>
    %1859 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1860 = stablehlo.add %1858, %1859 : tensor<1x8x1xf32>
    %1861 = stablehlo.rsqrt %1860 : tensor<1x8x1xf32>
    %1862 = stablehlo.broadcast_in_dim %1861, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %1863 = stablehlo.multiply %1853, %1862 : tensor<1x8x1024xf32>
    %1864 = stablehlo.convert %1863 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %1865 = stablehlo.convert %arg122 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %1866 = stablehlo.broadcast_in_dim %1865, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %1867 = stablehlo.broadcast_in_dim %1866, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1868 = stablehlo.multiply %1867, %1864 : tensor<1x8x1024xbf16>
    %1869 = stablehlo.convert %arg131 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %1870 = stablehlo.dot_general %1868, %1869, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %1871 = stablehlo.convert %arg128 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1872 = stablehlo.dot_general %1868, %1871, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1873 = stablehlo.convert %arg132 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %1874 = stablehlo.dot_general %1868, %1873, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1875 = stablehlo.reshape %1870 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %1876 = stablehlo.convert %1875 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %1877 = stablehlo.multiply %1876, %1876 : tensor<1x8x16x128xf32>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1878 = stablehlo.reduce(%1877 init: %cst_88) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %1880 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1881 = stablehlo.divide %1879, %1880 : tensor<1x8x16x1xf32>
    %1882 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %1883 = stablehlo.add %1881, %1882 : tensor<1x8x16x1xf32>
    %1884 = stablehlo.rsqrt %1883 : tensor<1x8x16x1xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %1886 = stablehlo.multiply %1876, %1885 : tensor<1x8x16x128xf32>
    %1887 = stablehlo.convert %1886 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %1888 = stablehlo.convert %arg130 : (tensor<128xf32>) -> tensor<128xbf16>
    %1889 = stablehlo.broadcast_in_dim %1888, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1890 = stablehlo.broadcast_in_dim %1889, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1891 = stablehlo.multiply %1890, %1887 : tensor<1x8x16x128xbf16>
    %1892 = stablehlo.reshape %1872 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1893 = stablehlo.convert %1892 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %1894 = stablehlo.multiply %1893, %1893 : tensor<1x8x8x128xf32>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1895 = stablehlo.reduce(%1894 init: %cst_89) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %1896 = stablehlo.broadcast_in_dim %1895, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %1897 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1898 = stablehlo.divide %1896, %1897 : tensor<1x8x8x1xf32>
    %1899 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %1900 = stablehlo.add %1898, %1899 : tensor<1x8x8x1xf32>
    %1901 = stablehlo.rsqrt %1900 : tensor<1x8x8x1xf32>
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %1903 = stablehlo.multiply %1893, %1902 : tensor<1x8x8x128xf32>
    %1904 = stablehlo.convert %1903 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %1905 = stablehlo.convert %arg127 : (tensor<128xf32>) -> tensor<128xbf16>
    %1906 = stablehlo.broadcast_in_dim %1905, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %1907 = stablehlo.broadcast_in_dim %1906, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %1908 = stablehlo.multiply %1907, %1904 : tensor<1x8x8x128xbf16>
    %1909 = stablehlo.reshape %1874 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %1910 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1911 = stablehlo.compare  LT, %7, %1910,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %1912 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %1913 = stablehlo.add %7, %1912 : tensor<1x8xi32>
    %1914 = stablehlo.select %1911, %1913, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %1915 = stablehlo.broadcast_in_dim %1914, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %1916 = "stablehlo.gather"(%26, %1915) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %1917 = stablehlo.slice %1916 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1918 = stablehlo.slice %1916 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %1919 = stablehlo.broadcast_in_dim %1917, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1920 = stablehlo.convert %1919 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1921 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1922 = stablehlo.convert %1921 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1923 = stablehlo.slice %1891 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1924 = stablehlo.slice %1891 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %1925 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1926 = stablehlo.multiply %1923, %1925 : tensor<1x8x16x64xbf16>
    %1927 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1928 = stablehlo.multiply %1924, %1927 : tensor<1x8x16x64xbf16>
    %1929 = stablehlo.subtract %1926, %1928 : tensor<1x8x16x64xbf16>
    %1930 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1931 = stablehlo.multiply %1924, %1930 : tensor<1x8x16x64xbf16>
    %1932 = stablehlo.broadcast_in_dim %1922, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %1933 = stablehlo.multiply %1923, %1932 : tensor<1x8x16x64xbf16>
    %1934 = stablehlo.add %1931, %1933 : tensor<1x8x16x64xbf16>
    %1935 = stablehlo.concatenate %1929, %1934, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %1936 = stablehlo.broadcast_in_dim %1917, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1937 = stablehlo.convert %1936 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1938 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %1939 = stablehlo.convert %1938 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %1940 = stablehlo.slice %1908 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1941 = stablehlo.slice %1908 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %1942 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1943 = stablehlo.multiply %1940, %1942 : tensor<1x8x8x64xbf16>
    %1944 = stablehlo.broadcast_in_dim %1939, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1945 = stablehlo.multiply %1941, %1944 : tensor<1x8x8x64xbf16>
    %1946 = stablehlo.subtract %1943, %1945 : tensor<1x8x8x64xbf16>
    %1947 = stablehlo.broadcast_in_dim %1937, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1948 = stablehlo.multiply %1941, %1947 : tensor<1x8x8x64xbf16>
    %1949 = stablehlo.broadcast_in_dim %1939, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %1950 = stablehlo.multiply %1940, %1949 : tensor<1x8x8x64xbf16>
    %1951 = stablehlo.add %1948, %1950 : tensor<1x8x8x64xbf16>
    %1952 = stablehlo.concatenate %1946, %1951, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %1953 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %1954 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %1955 = stablehlo.broadcast_in_dim %1953, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %1956 = stablehlo.and %1955, %1954 : tensor<1x1x8x8xi1>
    %1957 = stablehlo.convert %1956 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %1958 = sdy.sharding_constraint %1935 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1959 = sdy.sharding_constraint %1952 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1960 = sdy.sharding_constraint %1909 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %1961 = sdy.sharding_constraint %1957 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %1962 = stablehlo.reshape %1958 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1963 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %1964 = stablehlo.multiply %1962, %1963 : tensor<1x8x8x2x128xbf16>
    %1965 = stablehlo.dot_general %1959, %1964, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %1966 = stablehlo.transpose %1965, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1967 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %1968 = stablehlo.compare  NE, %1961, %1967,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %1969 = stablehlo.convert %1968 : tensor<1x1x8x8xi1>
    %1970 = stablehlo.broadcast_in_dim %1969, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %1971 = stablehlo.reshape %1970 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %1972 = call @_where_83(%1971, %1966, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %1973 = stablehlo.convert %1972 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_91 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1974 = stablehlo.reduce(%1973 init: %cst_91) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1975 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %1976 = stablehlo.maximum %1975, %1974 : tensor<1x8x2x8xf32>
    %1977 = stablehlo.broadcast_in_dim %1976, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1978 = stablehlo.broadcast_in_dim %1977, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1979 = stablehlo.subtract %1973, %1978 : tensor<1x8x2x8x8xf32>
    %1980 = stablehlo.exponential %1979 : tensor<1x8x2x8x8xf32>
    %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1981 = stablehlo.reduce(%1980 init: %cst_92) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %1982 = stablehlo.broadcast_in_dim %1981, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %1983 = stablehlo.broadcast_in_dim %1982, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %1984 = stablehlo.divide %1980, %1983 : tensor<1x8x2x8x8xf32>
    %1985 = stablehlo.convert %1984 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %1986 = stablehlo.dot_general %1960, %1985, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %1987 = stablehlo.transpose %1986, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %1988 = stablehlo.reshape %1987 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %1989 = sdy.sharding_constraint %1988 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %1990 = stablehlo.reshape %1989 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %1991 = stablehlo.convert %arg129 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %1992 = stablehlo.dot_general %1990, %1991, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %1993 = stablehlo.add %1852, %1992 : tensor<1x8x1024xbf16>
    %1994 = stablehlo.convert %1993 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %1995 = stablehlo.multiply %1994, %1994 : tensor<1x8x1024xf32>
    %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1996 = stablehlo.reduce(%1995 init: %cst_93) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %1997 = stablehlo.broadcast_in_dim %1996, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %1998 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %1999 = stablehlo.divide %1997, %1998 : tensor<1x8x1xf32>
    %2000 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2001 = stablehlo.add %1999, %2000 : tensor<1x8x1xf32>
    %2002 = stablehlo.rsqrt %2001 : tensor<1x8x1xf32>
    %2003 = stablehlo.broadcast_in_dim %2002, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2004 = stablehlo.multiply %1994, %2003 : tensor<1x8x1024xf32>
    %2005 = stablehlo.convert %2004 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2006 = stablehlo.convert %arg126 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2007 = stablehlo.broadcast_in_dim %2006, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2008 = stablehlo.broadcast_in_dim %2007, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2009 = stablehlo.multiply %2008, %2005 : tensor<1x8x1024xbf16>
    %2010 = stablehlo.convert %arg124 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2011 = stablehlo.dot_general %2009, %2010, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2012 = call @silu(%2011) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2013 = stablehlo.convert %arg125 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2014 = stablehlo.dot_general %2009, %2013, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2015 = stablehlo.multiply %2012, %2014 : tensor<1x8x3072xbf16>
    %2016 = stablehlo.convert %arg123 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2017 = stablehlo.dot_general %2015, %2016, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2018 = stablehlo.add %1993, %2017 : tensor<1x8x1024xbf16>
    %2019 = stablehlo.convert %2018 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2020 = stablehlo.multiply %2019, %2019 : tensor<1x8x1024xf32>
    %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2021 = stablehlo.reduce(%2020 init: %cst_94) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2022 = stablehlo.broadcast_in_dim %2021, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2023 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2024 = stablehlo.divide %2022, %2023 : tensor<1x8x1xf32>
    %2025 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2026 = stablehlo.add %2024, %2025 : tensor<1x8x1xf32>
    %2027 = stablehlo.rsqrt %2026 : tensor<1x8x1xf32>
    %2028 = stablehlo.broadcast_in_dim %2027, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2029 = stablehlo.multiply %2019, %2028 : tensor<1x8x1024xf32>
    %2030 = stablehlo.convert %2029 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2031 = stablehlo.convert %arg133 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2032 = stablehlo.broadcast_in_dim %2031, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2033 = stablehlo.broadcast_in_dim %2032, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2034 = stablehlo.multiply %2033, %2030 : tensor<1x8x1024xbf16>
    %2035 = stablehlo.convert %arg142 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2036 = stablehlo.dot_general %2034, %2035, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2037 = stablehlo.convert %arg139 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2038 = stablehlo.dot_general %2034, %2037, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2039 = stablehlo.convert %arg143 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2040 = stablehlo.dot_general %2034, %2039, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2041 = stablehlo.reshape %2036 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2042 = stablehlo.convert %2041 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2043 = stablehlo.multiply %2042, %2042 : tensor<1x8x16x128xf32>
    %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2044 = stablehlo.reduce(%2043 init: %cst_95) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2046 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2047 = stablehlo.divide %2045, %2046 : tensor<1x8x16x1xf32>
    %2048 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2049 = stablehlo.add %2047, %2048 : tensor<1x8x16x1xf32>
    %2050 = stablehlo.rsqrt %2049 : tensor<1x8x16x1xf32>
    %2051 = stablehlo.broadcast_in_dim %2050, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2052 = stablehlo.multiply %2042, %2051 : tensor<1x8x16x128xf32>
    %2053 = stablehlo.convert %2052 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2054 = stablehlo.convert %arg141 : (tensor<128xf32>) -> tensor<128xbf16>
    %2055 = stablehlo.broadcast_in_dim %2054, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2056 = stablehlo.broadcast_in_dim %2055, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2057 = stablehlo.multiply %2056, %2053 : tensor<1x8x16x128xbf16>
    %2058 = stablehlo.reshape %2038 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2059 = stablehlo.convert %2058 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2060 = stablehlo.multiply %2059, %2059 : tensor<1x8x8x128xf32>
    %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2061 = stablehlo.reduce(%2060 init: %cst_96) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2062 = stablehlo.broadcast_in_dim %2061, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2063 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2064 = stablehlo.divide %2062, %2063 : tensor<1x8x8x1xf32>
    %2065 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2066 = stablehlo.add %2064, %2065 : tensor<1x8x8x1xf32>
    %2067 = stablehlo.rsqrt %2066 : tensor<1x8x8x1xf32>
    %2068 = stablehlo.broadcast_in_dim %2067, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2069 = stablehlo.multiply %2059, %2068 : tensor<1x8x8x128xf32>
    %2070 = stablehlo.convert %2069 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2071 = stablehlo.convert %arg138 : (tensor<128xf32>) -> tensor<128xbf16>
    %2072 = stablehlo.broadcast_in_dim %2071, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2073 = stablehlo.broadcast_in_dim %2072, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2074 = stablehlo.multiply %2073, %2070 : tensor<1x8x8x128xbf16>
    %2075 = stablehlo.reshape %2040 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2076 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2077 = stablehlo.compare  LT, %7, %2076,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2078 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2079 = stablehlo.add %7, %2078 : tensor<1x8xi32>
    %2080 = stablehlo.select %2077, %2079, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2081 = stablehlo.broadcast_in_dim %2080, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2082 = "stablehlo.gather"(%26, %2081) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2083 = stablehlo.slice %2082 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2084 = stablehlo.slice %2082 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2085 = stablehlo.broadcast_in_dim %2083, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2086 = stablehlo.convert %2085 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2087 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2088 = stablehlo.convert %2087 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2089 = stablehlo.slice %2057 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2090 = stablehlo.slice %2057 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2091 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2092 = stablehlo.multiply %2089, %2091 : tensor<1x8x16x64xbf16>
    %2093 = stablehlo.broadcast_in_dim %2088, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2094 = stablehlo.multiply %2090, %2093 : tensor<1x8x16x64xbf16>
    %2095 = stablehlo.subtract %2092, %2094 : tensor<1x8x16x64xbf16>
    %2096 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2097 = stablehlo.multiply %2090, %2096 : tensor<1x8x16x64xbf16>
    %2098 = stablehlo.broadcast_in_dim %2088, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2099 = stablehlo.multiply %2089, %2098 : tensor<1x8x16x64xbf16>
    %2100 = stablehlo.add %2097, %2099 : tensor<1x8x16x64xbf16>
    %2101 = stablehlo.concatenate %2095, %2100, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2102 = stablehlo.broadcast_in_dim %2083, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2103 = stablehlo.convert %2102 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2104 = stablehlo.broadcast_in_dim %2084, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2105 = stablehlo.convert %2104 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2106 = stablehlo.slice %2074 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2107 = stablehlo.slice %2074 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2108 = stablehlo.broadcast_in_dim %2103, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2109 = stablehlo.multiply %2106, %2108 : tensor<1x8x8x64xbf16>
    %2110 = stablehlo.broadcast_in_dim %2105, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2111 = stablehlo.multiply %2107, %2110 : tensor<1x8x8x64xbf16>
    %2112 = stablehlo.subtract %2109, %2111 : tensor<1x8x8x64xbf16>
    %2113 = stablehlo.broadcast_in_dim %2103, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2114 = stablehlo.multiply %2107, %2113 : tensor<1x8x8x64xbf16>
    %2115 = stablehlo.broadcast_in_dim %2105, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2116 = stablehlo.multiply %2106, %2115 : tensor<1x8x8x64xbf16>
    %2117 = stablehlo.add %2114, %2116 : tensor<1x8x8x64xbf16>
    %2118 = stablehlo.concatenate %2112, %2117, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2119 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2120 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2121 = stablehlo.broadcast_in_dim %2119, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2122 = stablehlo.and %2121, %2120 : tensor<1x1x8x8xi1>
    %2123 = stablehlo.convert %2122 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2124 = sdy.sharding_constraint %2101 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2125 = sdy.sharding_constraint %2118 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2126 = sdy.sharding_constraint %2075 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2127 = sdy.sharding_constraint %2123 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2128 = stablehlo.reshape %2124 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2129 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2130 = stablehlo.multiply %2128, %2129 : tensor<1x8x8x2x128xbf16>
    %2131 = stablehlo.dot_general %2125, %2130, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2132 = stablehlo.transpose %2131, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2133 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2134 = stablehlo.compare  NE, %2127, %2133,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2135 = stablehlo.convert %2134 : tensor<1x1x8x8xi1>
    %2136 = stablehlo.broadcast_in_dim %2135, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2137 = stablehlo.reshape %2136 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2138 = call @_where_83(%2137, %2132, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2139 = stablehlo.convert %2138 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_98 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2140 = stablehlo.reduce(%2139 init: %cst_98) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2141 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2142 = stablehlo.maximum %2141, %2140 : tensor<1x8x2x8xf32>
    %2143 = stablehlo.broadcast_in_dim %2142, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2144 = stablehlo.broadcast_in_dim %2143, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2145 = stablehlo.subtract %2139, %2144 : tensor<1x8x2x8x8xf32>
    %2146 = stablehlo.exponential %2145 : tensor<1x8x2x8x8xf32>
    %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2147 = stablehlo.reduce(%2146 init: %cst_99) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2148 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2149 = stablehlo.broadcast_in_dim %2148, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2150 = stablehlo.divide %2146, %2149 : tensor<1x8x2x8x8xf32>
    %2151 = stablehlo.convert %2150 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2152 = stablehlo.dot_general %2126, %2151, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2153 = stablehlo.transpose %2152, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2154 = stablehlo.reshape %2153 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2155 = sdy.sharding_constraint %2154 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2156 = stablehlo.reshape %2155 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2157 = stablehlo.convert %arg140 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2158 = stablehlo.dot_general %2156, %2157, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2159 = stablehlo.add %2018, %2158 : tensor<1x8x1024xbf16>
    %2160 = stablehlo.convert %2159 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2161 = stablehlo.multiply %2160, %2160 : tensor<1x8x1024xf32>
    %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2162 = stablehlo.reduce(%2161 init: %cst_100) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2163 = stablehlo.broadcast_in_dim %2162, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2164 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2165 = stablehlo.divide %2163, %2164 : tensor<1x8x1xf32>
    %2166 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2167 = stablehlo.add %2165, %2166 : tensor<1x8x1xf32>
    %2168 = stablehlo.rsqrt %2167 : tensor<1x8x1xf32>
    %2169 = stablehlo.broadcast_in_dim %2168, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2170 = stablehlo.multiply %2160, %2169 : tensor<1x8x1024xf32>
    %2171 = stablehlo.convert %2170 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2172 = stablehlo.convert %arg137 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2173 = stablehlo.broadcast_in_dim %2172, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2174 = stablehlo.broadcast_in_dim %2173, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2175 = stablehlo.multiply %2174, %2171 : tensor<1x8x1024xbf16>
    %2176 = stablehlo.convert %arg135 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2177 = stablehlo.dot_general %2175, %2176, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2178 = call @silu(%2177) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2179 = stablehlo.convert %arg136 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2180 = stablehlo.dot_general %2175, %2179, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2181 = stablehlo.multiply %2178, %2180 : tensor<1x8x3072xbf16>
    %2182 = stablehlo.convert %arg134 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2183 = stablehlo.dot_general %2181, %2182, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2184 = stablehlo.add %2159, %2183 : tensor<1x8x1024xbf16>
    %2185 = stablehlo.convert %2184 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2186 = stablehlo.multiply %2185, %2185 : tensor<1x8x1024xf32>
    %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2187 = stablehlo.reduce(%2186 init: %cst_101) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2188 = stablehlo.broadcast_in_dim %2187, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2189 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2190 = stablehlo.divide %2188, %2189 : tensor<1x8x1xf32>
    %2191 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2192 = stablehlo.add %2190, %2191 : tensor<1x8x1xf32>
    %2193 = stablehlo.rsqrt %2192 : tensor<1x8x1xf32>
    %2194 = stablehlo.broadcast_in_dim %2193, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2195 = stablehlo.multiply %2185, %2194 : tensor<1x8x1024xf32>
    %2196 = stablehlo.convert %2195 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2197 = stablehlo.convert %arg144 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2198 = stablehlo.broadcast_in_dim %2197, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2199 = stablehlo.broadcast_in_dim %2198, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2200 = stablehlo.multiply %2199, %2196 : tensor<1x8x1024xbf16>
    %2201 = stablehlo.convert %arg153 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2202 = stablehlo.dot_general %2200, %2201, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2203 = stablehlo.convert %arg150 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2204 = stablehlo.dot_general %2200, %2203, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2205 = stablehlo.convert %arg154 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2206 = stablehlo.dot_general %2200, %2205, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2207 = stablehlo.reshape %2202 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2208 = stablehlo.convert %2207 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2209 = stablehlo.multiply %2208, %2208 : tensor<1x8x16x128xf32>
    %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2210 = stablehlo.reduce(%2209 init: %cst_102) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2211 = stablehlo.broadcast_in_dim %2210, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2212 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2213 = stablehlo.divide %2211, %2212 : tensor<1x8x16x1xf32>
    %2214 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2215 = stablehlo.add %2213, %2214 : tensor<1x8x16x1xf32>
    %2216 = stablehlo.rsqrt %2215 : tensor<1x8x16x1xf32>
    %2217 = stablehlo.broadcast_in_dim %2216, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2218 = stablehlo.multiply %2208, %2217 : tensor<1x8x16x128xf32>
    %2219 = stablehlo.convert %2218 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2220 = stablehlo.convert %arg152 : (tensor<128xf32>) -> tensor<128xbf16>
    %2221 = stablehlo.broadcast_in_dim %2220, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2222 = stablehlo.broadcast_in_dim %2221, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2223 = stablehlo.multiply %2222, %2219 : tensor<1x8x16x128xbf16>
    %2224 = stablehlo.reshape %2204 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2225 = stablehlo.convert %2224 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2226 = stablehlo.multiply %2225, %2225 : tensor<1x8x8x128xf32>
    %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2227 = stablehlo.reduce(%2226 init: %cst_103) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2228 = stablehlo.broadcast_in_dim %2227, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2229 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2230 = stablehlo.divide %2228, %2229 : tensor<1x8x8x1xf32>
    %2231 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2232 = stablehlo.add %2230, %2231 : tensor<1x8x8x1xf32>
    %2233 = stablehlo.rsqrt %2232 : tensor<1x8x8x1xf32>
    %2234 = stablehlo.broadcast_in_dim %2233, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2235 = stablehlo.multiply %2225, %2234 : tensor<1x8x8x128xf32>
    %2236 = stablehlo.convert %2235 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2237 = stablehlo.convert %arg149 : (tensor<128xf32>) -> tensor<128xbf16>
    %2238 = stablehlo.broadcast_in_dim %2237, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2239 = stablehlo.broadcast_in_dim %2238, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2240 = stablehlo.multiply %2239, %2236 : tensor<1x8x8x128xbf16>
    %2241 = stablehlo.reshape %2206 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2242 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2243 = stablehlo.compare  LT, %7, %2242,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2244 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2245 = stablehlo.add %7, %2244 : tensor<1x8xi32>
    %2246 = stablehlo.select %2243, %2245, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2247 = stablehlo.broadcast_in_dim %2246, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2248 = "stablehlo.gather"(%26, %2247) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2249 = stablehlo.slice %2248 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2250 = stablehlo.slice %2248 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2251 = stablehlo.broadcast_in_dim %2249, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2252 = stablehlo.convert %2251 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2253 = stablehlo.broadcast_in_dim %2250, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2254 = stablehlo.convert %2253 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2255 = stablehlo.slice %2223 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2256 = stablehlo.slice %2223 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2257 = stablehlo.broadcast_in_dim %2252, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2258 = stablehlo.multiply %2255, %2257 : tensor<1x8x16x64xbf16>
    %2259 = stablehlo.broadcast_in_dim %2254, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2260 = stablehlo.multiply %2256, %2259 : tensor<1x8x16x64xbf16>
    %2261 = stablehlo.subtract %2258, %2260 : tensor<1x8x16x64xbf16>
    %2262 = stablehlo.broadcast_in_dim %2252, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2263 = stablehlo.multiply %2256, %2262 : tensor<1x8x16x64xbf16>
    %2264 = stablehlo.broadcast_in_dim %2254, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2265 = stablehlo.multiply %2255, %2264 : tensor<1x8x16x64xbf16>
    %2266 = stablehlo.add %2263, %2265 : tensor<1x8x16x64xbf16>
    %2267 = stablehlo.concatenate %2261, %2266, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2268 = stablehlo.broadcast_in_dim %2249, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2269 = stablehlo.convert %2268 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2270 = stablehlo.broadcast_in_dim %2250, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2271 = stablehlo.convert %2270 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2272 = stablehlo.slice %2240 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2273 = stablehlo.slice %2240 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2274 = stablehlo.broadcast_in_dim %2269, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2275 = stablehlo.multiply %2272, %2274 : tensor<1x8x8x64xbf16>
    %2276 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2277 = stablehlo.multiply %2273, %2276 : tensor<1x8x8x64xbf16>
    %2278 = stablehlo.subtract %2275, %2277 : tensor<1x8x8x64xbf16>
    %2279 = stablehlo.broadcast_in_dim %2269, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2280 = stablehlo.multiply %2273, %2279 : tensor<1x8x8x64xbf16>
    %2281 = stablehlo.broadcast_in_dim %2271, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2282 = stablehlo.multiply %2272, %2281 : tensor<1x8x8x64xbf16>
    %2283 = stablehlo.add %2280, %2282 : tensor<1x8x8x64xbf16>
    %2284 = stablehlo.concatenate %2278, %2283, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2285 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2286 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2287 = stablehlo.broadcast_in_dim %2285, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2288 = stablehlo.and %2287, %2286 : tensor<1x1x8x8xi1>
    %2289 = stablehlo.convert %2288 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2290 = sdy.sharding_constraint %2267 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2291 = sdy.sharding_constraint %2284 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2292 = sdy.sharding_constraint %2241 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2293 = sdy.sharding_constraint %2289 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2294 = stablehlo.reshape %2290 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2295 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2296 = stablehlo.multiply %2294, %2295 : tensor<1x8x8x2x128xbf16>
    %2297 = stablehlo.dot_general %2291, %2296, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2298 = stablehlo.transpose %2297, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2299 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2300 = stablehlo.compare  NE, %2293, %2299,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2301 = stablehlo.convert %2300 : tensor<1x1x8x8xi1>
    %2302 = stablehlo.broadcast_in_dim %2301, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2303 = stablehlo.reshape %2302 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2304 = call @_where_83(%2303, %2298, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2305 = stablehlo.convert %2304 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_105 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2306 = stablehlo.reduce(%2305 init: %cst_105) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2307 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2308 = stablehlo.maximum %2307, %2306 : tensor<1x8x2x8xf32>
    %2309 = stablehlo.broadcast_in_dim %2308, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2311 = stablehlo.subtract %2305, %2310 : tensor<1x8x2x8x8xf32>
    %2312 = stablehlo.exponential %2311 : tensor<1x8x2x8x8xf32>
    %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2313 = stablehlo.reduce(%2312 init: %cst_106) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2314 = stablehlo.broadcast_in_dim %2313, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2315 = stablehlo.broadcast_in_dim %2314, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2316 = stablehlo.divide %2312, %2315 : tensor<1x8x2x8x8xf32>
    %2317 = stablehlo.convert %2316 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2318 = stablehlo.dot_general %2292, %2317, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2319 = stablehlo.transpose %2318, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2320 = stablehlo.reshape %2319 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2321 = sdy.sharding_constraint %2320 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2322 = stablehlo.reshape %2321 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2323 = stablehlo.convert %arg151 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2324 = stablehlo.dot_general %2322, %2323, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2325 = stablehlo.add %2184, %2324 : tensor<1x8x1024xbf16>
    %2326 = stablehlo.convert %2325 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2327 = stablehlo.multiply %2326, %2326 : tensor<1x8x1024xf32>
    %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2328 = stablehlo.reduce(%2327 init: %cst_107) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2329 = stablehlo.broadcast_in_dim %2328, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2330 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2331 = stablehlo.divide %2329, %2330 : tensor<1x8x1xf32>
    %2332 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2333 = stablehlo.add %2331, %2332 : tensor<1x8x1xf32>
    %2334 = stablehlo.rsqrt %2333 : tensor<1x8x1xf32>
    %2335 = stablehlo.broadcast_in_dim %2334, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2336 = stablehlo.multiply %2326, %2335 : tensor<1x8x1024xf32>
    %2337 = stablehlo.convert %2336 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2338 = stablehlo.convert %arg148 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2339 = stablehlo.broadcast_in_dim %2338, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2340 = stablehlo.broadcast_in_dim %2339, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2341 = stablehlo.multiply %2340, %2337 : tensor<1x8x1024xbf16>
    %2342 = stablehlo.convert %arg146 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2343 = stablehlo.dot_general %2341, %2342, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2344 = call @silu(%2343) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2345 = stablehlo.convert %arg147 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2346 = stablehlo.dot_general %2341, %2345, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2347 = stablehlo.multiply %2344, %2346 : tensor<1x8x3072xbf16>
    %2348 = stablehlo.convert %arg145 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2349 = stablehlo.dot_general %2347, %2348, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2350 = stablehlo.add %2325, %2349 : tensor<1x8x1024xbf16>
    %2351 = stablehlo.convert %2350 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2352 = stablehlo.multiply %2351, %2351 : tensor<1x8x1024xf32>
    %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2353 = stablehlo.reduce(%2352 init: %cst_108) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2354 = stablehlo.broadcast_in_dim %2353, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2355 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2356 = stablehlo.divide %2354, %2355 : tensor<1x8x1xf32>
    %2357 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2358 = stablehlo.add %2356, %2357 : tensor<1x8x1xf32>
    %2359 = stablehlo.rsqrt %2358 : tensor<1x8x1xf32>
    %2360 = stablehlo.broadcast_in_dim %2359, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2361 = stablehlo.multiply %2351, %2360 : tensor<1x8x1024xf32>
    %2362 = stablehlo.convert %2361 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2363 = stablehlo.convert %arg155 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2364 = stablehlo.broadcast_in_dim %2363, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2365 = stablehlo.broadcast_in_dim %2364, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2366 = stablehlo.multiply %2365, %2362 : tensor<1x8x1024xbf16>
    %2367 = stablehlo.convert %arg164 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2368 = stablehlo.dot_general %2366, %2367, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2369 = stablehlo.convert %arg161 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2370 = stablehlo.dot_general %2366, %2369, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2371 = stablehlo.convert %arg165 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2372 = stablehlo.dot_general %2366, %2371, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2373 = stablehlo.reshape %2368 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2374 = stablehlo.convert %2373 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2375 = stablehlo.multiply %2374, %2374 : tensor<1x8x16x128xf32>
    %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2376 = stablehlo.reduce(%2375 init: %cst_109) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2377 = stablehlo.broadcast_in_dim %2376, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2378 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2379 = stablehlo.divide %2377, %2378 : tensor<1x8x16x1xf32>
    %2380 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2381 = stablehlo.add %2379, %2380 : tensor<1x8x16x1xf32>
    %2382 = stablehlo.rsqrt %2381 : tensor<1x8x16x1xf32>
    %2383 = stablehlo.broadcast_in_dim %2382, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2384 = stablehlo.multiply %2374, %2383 : tensor<1x8x16x128xf32>
    %2385 = stablehlo.convert %2384 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2386 = stablehlo.convert %arg163 : (tensor<128xf32>) -> tensor<128xbf16>
    %2387 = stablehlo.broadcast_in_dim %2386, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2388 = stablehlo.broadcast_in_dim %2387, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2389 = stablehlo.multiply %2388, %2385 : tensor<1x8x16x128xbf16>
    %2390 = stablehlo.reshape %2370 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2391 = stablehlo.convert %2390 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2392 = stablehlo.multiply %2391, %2391 : tensor<1x8x8x128xf32>
    %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2393 = stablehlo.reduce(%2392 init: %cst_110) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2394 = stablehlo.broadcast_in_dim %2393, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2395 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2396 = stablehlo.divide %2394, %2395 : tensor<1x8x8x1xf32>
    %2397 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2398 = stablehlo.add %2396, %2397 : tensor<1x8x8x1xf32>
    %2399 = stablehlo.rsqrt %2398 : tensor<1x8x8x1xf32>
    %2400 = stablehlo.broadcast_in_dim %2399, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2401 = stablehlo.multiply %2391, %2400 : tensor<1x8x8x128xf32>
    %2402 = stablehlo.convert %2401 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2403 = stablehlo.convert %arg160 : (tensor<128xf32>) -> tensor<128xbf16>
    %2404 = stablehlo.broadcast_in_dim %2403, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2405 = stablehlo.broadcast_in_dim %2404, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2406 = stablehlo.multiply %2405, %2402 : tensor<1x8x8x128xbf16>
    %2407 = stablehlo.reshape %2372 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2408 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2409 = stablehlo.compare  LT, %7, %2408,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2410 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2411 = stablehlo.add %7, %2410 : tensor<1x8xi32>
    %2412 = stablehlo.select %2409, %2411, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2413 = stablehlo.broadcast_in_dim %2412, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2414 = "stablehlo.gather"(%26, %2413) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2415 = stablehlo.slice %2414 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2416 = stablehlo.slice %2414 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2417 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2418 = stablehlo.convert %2417 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2419 = stablehlo.broadcast_in_dim %2416, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2420 = stablehlo.convert %2419 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2421 = stablehlo.slice %2389 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2422 = stablehlo.slice %2389 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2423 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2424 = stablehlo.multiply %2421, %2423 : tensor<1x8x16x64xbf16>
    %2425 = stablehlo.broadcast_in_dim %2420, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2426 = stablehlo.multiply %2422, %2425 : tensor<1x8x16x64xbf16>
    %2427 = stablehlo.subtract %2424, %2426 : tensor<1x8x16x64xbf16>
    %2428 = stablehlo.broadcast_in_dim %2418, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2429 = stablehlo.multiply %2422, %2428 : tensor<1x8x16x64xbf16>
    %2430 = stablehlo.broadcast_in_dim %2420, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2431 = stablehlo.multiply %2421, %2430 : tensor<1x8x16x64xbf16>
    %2432 = stablehlo.add %2429, %2431 : tensor<1x8x16x64xbf16>
    %2433 = stablehlo.concatenate %2427, %2432, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2434 = stablehlo.broadcast_in_dim %2415, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2435 = stablehlo.convert %2434 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2436 = stablehlo.broadcast_in_dim %2416, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2437 = stablehlo.convert %2436 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2438 = stablehlo.slice %2406 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2439 = stablehlo.slice %2406 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2440 = stablehlo.broadcast_in_dim %2435, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2441 = stablehlo.multiply %2438, %2440 : tensor<1x8x8x64xbf16>
    %2442 = stablehlo.broadcast_in_dim %2437, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2443 = stablehlo.multiply %2439, %2442 : tensor<1x8x8x64xbf16>
    %2444 = stablehlo.subtract %2441, %2443 : tensor<1x8x8x64xbf16>
    %2445 = stablehlo.broadcast_in_dim %2435, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2446 = stablehlo.multiply %2439, %2445 : tensor<1x8x8x64xbf16>
    %2447 = stablehlo.broadcast_in_dim %2437, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2448 = stablehlo.multiply %2438, %2447 : tensor<1x8x8x64xbf16>
    %2449 = stablehlo.add %2446, %2448 : tensor<1x8x8x64xbf16>
    %2450 = stablehlo.concatenate %2444, %2449, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2451 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2452 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2453 = stablehlo.broadcast_in_dim %2451, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2454 = stablehlo.and %2453, %2452 : tensor<1x1x8x8xi1>
    %2455 = stablehlo.convert %2454 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2456 = sdy.sharding_constraint %2433 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2457 = sdy.sharding_constraint %2450 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2458 = sdy.sharding_constraint %2407 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2459 = sdy.sharding_constraint %2455 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2460 = stablehlo.reshape %2456 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2461 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2462 = stablehlo.multiply %2460, %2461 : tensor<1x8x8x2x128xbf16>
    %2463 = stablehlo.dot_general %2457, %2462, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2464 = stablehlo.transpose %2463, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2465 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2466 = stablehlo.compare  NE, %2459, %2465,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2467 = stablehlo.convert %2466 : tensor<1x1x8x8xi1>
    %2468 = stablehlo.broadcast_in_dim %2467, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2469 = stablehlo.reshape %2468 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2470 = call @_where_83(%2469, %2464, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2471 = stablehlo.convert %2470 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_112 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2472 = stablehlo.reduce(%2471 init: %cst_112) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2473 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2474 = stablehlo.maximum %2473, %2472 : tensor<1x8x2x8xf32>
    %2475 = stablehlo.broadcast_in_dim %2474, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2477 = stablehlo.subtract %2471, %2476 : tensor<1x8x2x8x8xf32>
    %2478 = stablehlo.exponential %2477 : tensor<1x8x2x8x8xf32>
    %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2479 = stablehlo.reduce(%2478 init: %cst_113) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2480 = stablehlo.broadcast_in_dim %2479, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2481 = stablehlo.broadcast_in_dim %2480, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2482 = stablehlo.divide %2478, %2481 : tensor<1x8x2x8x8xf32>
    %2483 = stablehlo.convert %2482 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2484 = stablehlo.dot_general %2458, %2483, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2485 = stablehlo.transpose %2484, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2486 = stablehlo.reshape %2485 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2487 = sdy.sharding_constraint %2486 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2488 = stablehlo.reshape %2487 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2489 = stablehlo.convert %arg162 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2490 = stablehlo.dot_general %2488, %2489, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2491 = stablehlo.add %2350, %2490 : tensor<1x8x1024xbf16>
    %2492 = stablehlo.convert %2491 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2493 = stablehlo.multiply %2492, %2492 : tensor<1x8x1024xf32>
    %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2494 = stablehlo.reduce(%2493 init: %cst_114) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2495 = stablehlo.broadcast_in_dim %2494, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2496 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2497 = stablehlo.divide %2495, %2496 : tensor<1x8x1xf32>
    %2498 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2499 = stablehlo.add %2497, %2498 : tensor<1x8x1xf32>
    %2500 = stablehlo.rsqrt %2499 : tensor<1x8x1xf32>
    %2501 = stablehlo.broadcast_in_dim %2500, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2502 = stablehlo.multiply %2492, %2501 : tensor<1x8x1024xf32>
    %2503 = stablehlo.convert %2502 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2504 = stablehlo.convert %arg159 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2505 = stablehlo.broadcast_in_dim %2504, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2506 = stablehlo.broadcast_in_dim %2505, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2507 = stablehlo.multiply %2506, %2503 : tensor<1x8x1024xbf16>
    %2508 = stablehlo.convert %arg157 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2509 = stablehlo.dot_general %2507, %2508, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2510 = call @silu(%2509) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2511 = stablehlo.convert %arg158 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2512 = stablehlo.dot_general %2507, %2511, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2513 = stablehlo.multiply %2510, %2512 : tensor<1x8x3072xbf16>
    %2514 = stablehlo.convert %arg156 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2515 = stablehlo.dot_general %2513, %2514, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2516 = stablehlo.add %2491, %2515 : tensor<1x8x1024xbf16>
    %2517 = stablehlo.convert %2516 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2518 = stablehlo.multiply %2517, %2517 : tensor<1x8x1024xf32>
    %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2519 = stablehlo.reduce(%2518 init: %cst_115) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2520 = stablehlo.broadcast_in_dim %2519, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2521 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2522 = stablehlo.divide %2520, %2521 : tensor<1x8x1xf32>
    %2523 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2524 = stablehlo.add %2522, %2523 : tensor<1x8x1xf32>
    %2525 = stablehlo.rsqrt %2524 : tensor<1x8x1xf32>
    %2526 = stablehlo.broadcast_in_dim %2525, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2527 = stablehlo.multiply %2517, %2526 : tensor<1x8x1024xf32>
    %2528 = stablehlo.convert %2527 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2529 = stablehlo.convert %arg166 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2530 = stablehlo.broadcast_in_dim %2529, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2531 = stablehlo.broadcast_in_dim %2530, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2532 = stablehlo.multiply %2531, %2528 : tensor<1x8x1024xbf16>
    %2533 = stablehlo.convert %arg175 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2534 = stablehlo.dot_general %2532, %2533, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2535 = stablehlo.convert %arg172 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2536 = stablehlo.dot_general %2532, %2535, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2537 = stablehlo.convert %arg176 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2538 = stablehlo.dot_general %2532, %2537, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2539 = stablehlo.reshape %2534 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2540 = stablehlo.convert %2539 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2541 = stablehlo.multiply %2540, %2540 : tensor<1x8x16x128xf32>
    %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2542 = stablehlo.reduce(%2541 init: %cst_116) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2543 = stablehlo.broadcast_in_dim %2542, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2544 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2545 = stablehlo.divide %2543, %2544 : tensor<1x8x16x1xf32>
    %2546 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2547 = stablehlo.add %2545, %2546 : tensor<1x8x16x1xf32>
    %2548 = stablehlo.rsqrt %2547 : tensor<1x8x16x1xf32>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2550 = stablehlo.multiply %2540, %2549 : tensor<1x8x16x128xf32>
    %2551 = stablehlo.convert %2550 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2552 = stablehlo.convert %arg174 : (tensor<128xf32>) -> tensor<128xbf16>
    %2553 = stablehlo.broadcast_in_dim %2552, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2554 = stablehlo.broadcast_in_dim %2553, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2555 = stablehlo.multiply %2554, %2551 : tensor<1x8x16x128xbf16>
    %2556 = stablehlo.reshape %2536 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2557 = stablehlo.convert %2556 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2558 = stablehlo.multiply %2557, %2557 : tensor<1x8x8x128xf32>
    %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2559 = stablehlo.reduce(%2558 init: %cst_117) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2560 = stablehlo.broadcast_in_dim %2559, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2561 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2562 = stablehlo.divide %2560, %2561 : tensor<1x8x8x1xf32>
    %2563 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2564 = stablehlo.add %2562, %2563 : tensor<1x8x8x1xf32>
    %2565 = stablehlo.rsqrt %2564 : tensor<1x8x8x1xf32>
    %2566 = stablehlo.broadcast_in_dim %2565, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2567 = stablehlo.multiply %2557, %2566 : tensor<1x8x8x128xf32>
    %2568 = stablehlo.convert %2567 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2569 = stablehlo.convert %arg171 : (tensor<128xf32>) -> tensor<128xbf16>
    %2570 = stablehlo.broadcast_in_dim %2569, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2571 = stablehlo.broadcast_in_dim %2570, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2572 = stablehlo.multiply %2571, %2568 : tensor<1x8x8x128xbf16>
    %2573 = stablehlo.reshape %2538 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2574 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2575 = stablehlo.compare  LT, %7, %2574,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2576 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2577 = stablehlo.add %7, %2576 : tensor<1x8xi32>
    %2578 = stablehlo.select %2575, %2577, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2579 = stablehlo.broadcast_in_dim %2578, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2580 = "stablehlo.gather"(%26, %2579) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2581 = stablehlo.slice %2580 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2582 = stablehlo.slice %2580 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2583 = stablehlo.broadcast_in_dim %2581, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2584 = stablehlo.convert %2583 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2585 = stablehlo.broadcast_in_dim %2582, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2586 = stablehlo.convert %2585 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2587 = stablehlo.slice %2555 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2588 = stablehlo.slice %2555 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2589 = stablehlo.broadcast_in_dim %2584, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2590 = stablehlo.multiply %2587, %2589 : tensor<1x8x16x64xbf16>
    %2591 = stablehlo.broadcast_in_dim %2586, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2592 = stablehlo.multiply %2588, %2591 : tensor<1x8x16x64xbf16>
    %2593 = stablehlo.subtract %2590, %2592 : tensor<1x8x16x64xbf16>
    %2594 = stablehlo.broadcast_in_dim %2584, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2595 = stablehlo.multiply %2588, %2594 : tensor<1x8x16x64xbf16>
    %2596 = stablehlo.broadcast_in_dim %2586, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2597 = stablehlo.multiply %2587, %2596 : tensor<1x8x16x64xbf16>
    %2598 = stablehlo.add %2595, %2597 : tensor<1x8x16x64xbf16>
    %2599 = stablehlo.concatenate %2593, %2598, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2600 = stablehlo.broadcast_in_dim %2581, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2601 = stablehlo.convert %2600 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2602 = stablehlo.broadcast_in_dim %2582, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2603 = stablehlo.convert %2602 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2604 = stablehlo.slice %2572 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2605 = stablehlo.slice %2572 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2606 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2607 = stablehlo.multiply %2604, %2606 : tensor<1x8x8x64xbf16>
    %2608 = stablehlo.broadcast_in_dim %2603, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2609 = stablehlo.multiply %2605, %2608 : tensor<1x8x8x64xbf16>
    %2610 = stablehlo.subtract %2607, %2609 : tensor<1x8x8x64xbf16>
    %2611 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2612 = stablehlo.multiply %2605, %2611 : tensor<1x8x8x64xbf16>
    %2613 = stablehlo.broadcast_in_dim %2603, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2614 = stablehlo.multiply %2604, %2613 : tensor<1x8x8x64xbf16>
    %2615 = stablehlo.add %2612, %2614 : tensor<1x8x8x64xbf16>
    %2616 = stablehlo.concatenate %2610, %2615, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2617 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2618 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2619 = stablehlo.broadcast_in_dim %2617, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2620 = stablehlo.and %2619, %2618 : tensor<1x1x8x8xi1>
    %2621 = stablehlo.convert %2620 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2622 = sdy.sharding_constraint %2599 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2623 = sdy.sharding_constraint %2616 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2624 = sdy.sharding_constraint %2573 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2625 = sdy.sharding_constraint %2621 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2626 = stablehlo.reshape %2622 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2627 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2628 = stablehlo.multiply %2626, %2627 : tensor<1x8x8x2x128xbf16>
    %2629 = stablehlo.dot_general %2623, %2628, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2630 = stablehlo.transpose %2629, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2631 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2632 = stablehlo.compare  NE, %2625, %2631,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2633 = stablehlo.convert %2632 : tensor<1x1x8x8xi1>
    %2634 = stablehlo.broadcast_in_dim %2633, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2635 = stablehlo.reshape %2634 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2636 = call @_where_83(%2635, %2630, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2637 = stablehlo.convert %2636 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_119 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2638 = stablehlo.reduce(%2637 init: %cst_119) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2639 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2640 = stablehlo.maximum %2639, %2638 : tensor<1x8x2x8xf32>
    %2641 = stablehlo.broadcast_in_dim %2640, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2642 = stablehlo.broadcast_in_dim %2641, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2643 = stablehlo.subtract %2637, %2642 : tensor<1x8x2x8x8xf32>
    %2644 = stablehlo.exponential %2643 : tensor<1x8x2x8x8xf32>
    %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2645 = stablehlo.reduce(%2644 init: %cst_120) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2646 = stablehlo.broadcast_in_dim %2645, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2648 = stablehlo.divide %2644, %2647 : tensor<1x8x2x8x8xf32>
    %2649 = stablehlo.convert %2648 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2650 = stablehlo.dot_general %2624, %2649, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2651 = stablehlo.transpose %2650, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2652 = stablehlo.reshape %2651 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2653 = sdy.sharding_constraint %2652 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2654 = stablehlo.reshape %2653 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2655 = stablehlo.convert %arg173 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2656 = stablehlo.dot_general %2654, %2655, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2657 = stablehlo.add %2516, %2656 : tensor<1x8x1024xbf16>
    %2658 = stablehlo.convert %2657 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2659 = stablehlo.multiply %2658, %2658 : tensor<1x8x1024xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2660 = stablehlo.reduce(%2659 init: %cst_121) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2661 = stablehlo.broadcast_in_dim %2660, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2662 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2663 = stablehlo.divide %2661, %2662 : tensor<1x8x1xf32>
    %2664 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2665 = stablehlo.add %2663, %2664 : tensor<1x8x1xf32>
    %2666 = stablehlo.rsqrt %2665 : tensor<1x8x1xf32>
    %2667 = stablehlo.broadcast_in_dim %2666, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2668 = stablehlo.multiply %2658, %2667 : tensor<1x8x1024xf32>
    %2669 = stablehlo.convert %2668 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2670 = stablehlo.convert %arg170 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2671 = stablehlo.broadcast_in_dim %2670, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2672 = stablehlo.broadcast_in_dim %2671, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2673 = stablehlo.multiply %2672, %2669 : tensor<1x8x1024xbf16>
    %2674 = stablehlo.convert %arg168 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2675 = stablehlo.dot_general %2673, %2674, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2676 = call @silu(%2675) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2677 = stablehlo.convert %arg169 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2678 = stablehlo.dot_general %2673, %2677, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2679 = stablehlo.multiply %2676, %2678 : tensor<1x8x3072xbf16>
    %2680 = stablehlo.convert %arg167 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2681 = stablehlo.dot_general %2679, %2680, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2682 = stablehlo.add %2657, %2681 : tensor<1x8x1024xbf16>
    %2683 = stablehlo.convert %2682 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2684 = stablehlo.multiply %2683, %2683 : tensor<1x8x1024xf32>
    %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2685 = stablehlo.reduce(%2684 init: %cst_122) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2686 = stablehlo.broadcast_in_dim %2685, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2687 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2688 = stablehlo.divide %2686, %2687 : tensor<1x8x1xf32>
    %2689 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2690 = stablehlo.add %2688, %2689 : tensor<1x8x1xf32>
    %2691 = stablehlo.rsqrt %2690 : tensor<1x8x1xf32>
    %2692 = stablehlo.broadcast_in_dim %2691, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2693 = stablehlo.multiply %2683, %2692 : tensor<1x8x1024xf32>
    %2694 = stablehlo.convert %2693 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2695 = stablehlo.convert %arg177 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2696 = stablehlo.broadcast_in_dim %2695, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2697 = stablehlo.broadcast_in_dim %2696, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2698 = stablehlo.multiply %2697, %2694 : tensor<1x8x1024xbf16>
    %2699 = stablehlo.convert %arg186 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2700 = stablehlo.dot_general %2698, %2699, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2701 = stablehlo.convert %arg183 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2702 = stablehlo.dot_general %2698, %2701, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2703 = stablehlo.convert %arg187 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2704 = stablehlo.dot_general %2698, %2703, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2705 = stablehlo.reshape %2700 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2706 = stablehlo.convert %2705 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2707 = stablehlo.multiply %2706, %2706 : tensor<1x8x16x128xf32>
    %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2708 = stablehlo.reduce(%2707 init: %cst_123) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2709 = stablehlo.broadcast_in_dim %2708, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2710 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2711 = stablehlo.divide %2709, %2710 : tensor<1x8x16x1xf32>
    %2712 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2713 = stablehlo.add %2711, %2712 : tensor<1x8x16x1xf32>
    %2714 = stablehlo.rsqrt %2713 : tensor<1x8x16x1xf32>
    %2715 = stablehlo.broadcast_in_dim %2714, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2716 = stablehlo.multiply %2706, %2715 : tensor<1x8x16x128xf32>
    %2717 = stablehlo.convert %2716 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2718 = stablehlo.convert %arg185 : (tensor<128xf32>) -> tensor<128xbf16>
    %2719 = stablehlo.broadcast_in_dim %2718, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2720 = stablehlo.broadcast_in_dim %2719, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2721 = stablehlo.multiply %2720, %2717 : tensor<1x8x16x128xbf16>
    %2722 = stablehlo.reshape %2702 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2723 = stablehlo.convert %2722 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2724 = stablehlo.multiply %2723, %2723 : tensor<1x8x8x128xf32>
    %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2725 = stablehlo.reduce(%2724 init: %cst_124) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2726 = stablehlo.broadcast_in_dim %2725, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2727 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2728 = stablehlo.divide %2726, %2727 : tensor<1x8x8x1xf32>
    %2729 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2730 = stablehlo.add %2728, %2729 : tensor<1x8x8x1xf32>
    %2731 = stablehlo.rsqrt %2730 : tensor<1x8x8x1xf32>
    %2732 = stablehlo.broadcast_in_dim %2731, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2733 = stablehlo.multiply %2723, %2732 : tensor<1x8x8x128xf32>
    %2734 = stablehlo.convert %2733 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2735 = stablehlo.convert %arg182 : (tensor<128xf32>) -> tensor<128xbf16>
    %2736 = stablehlo.broadcast_in_dim %2735, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2737 = stablehlo.broadcast_in_dim %2736, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2738 = stablehlo.multiply %2737, %2734 : tensor<1x8x8x128xbf16>
    %2739 = stablehlo.reshape %2704 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2740 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2741 = stablehlo.compare  LT, %7, %2740,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2742 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2743 = stablehlo.add %7, %2742 : tensor<1x8xi32>
    %2744 = stablehlo.select %2741, %2743, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2745 = stablehlo.broadcast_in_dim %2744, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2746 = "stablehlo.gather"(%26, %2745) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2747 = stablehlo.slice %2746 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2748 = stablehlo.slice %2746 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2749 = stablehlo.broadcast_in_dim %2747, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2750 = stablehlo.convert %2749 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2751 = stablehlo.broadcast_in_dim %2748, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2752 = stablehlo.convert %2751 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2753 = stablehlo.slice %2721 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2754 = stablehlo.slice %2721 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2755 = stablehlo.broadcast_in_dim %2750, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2756 = stablehlo.multiply %2753, %2755 : tensor<1x8x16x64xbf16>
    %2757 = stablehlo.broadcast_in_dim %2752, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2758 = stablehlo.multiply %2754, %2757 : tensor<1x8x16x64xbf16>
    %2759 = stablehlo.subtract %2756, %2758 : tensor<1x8x16x64xbf16>
    %2760 = stablehlo.broadcast_in_dim %2750, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2761 = stablehlo.multiply %2754, %2760 : tensor<1x8x16x64xbf16>
    %2762 = stablehlo.broadcast_in_dim %2752, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2763 = stablehlo.multiply %2753, %2762 : tensor<1x8x16x64xbf16>
    %2764 = stablehlo.add %2761, %2763 : tensor<1x8x16x64xbf16>
    %2765 = stablehlo.concatenate %2759, %2764, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2766 = stablehlo.broadcast_in_dim %2747, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2767 = stablehlo.convert %2766 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2768 = stablehlo.broadcast_in_dim %2748, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2769 = stablehlo.convert %2768 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2770 = stablehlo.slice %2738 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2771 = stablehlo.slice %2738 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2772 = stablehlo.broadcast_in_dim %2767, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2773 = stablehlo.multiply %2770, %2772 : tensor<1x8x8x64xbf16>
    %2774 = stablehlo.broadcast_in_dim %2769, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2775 = stablehlo.multiply %2771, %2774 : tensor<1x8x8x64xbf16>
    %2776 = stablehlo.subtract %2773, %2775 : tensor<1x8x8x64xbf16>
    %2777 = stablehlo.broadcast_in_dim %2767, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2778 = stablehlo.multiply %2771, %2777 : tensor<1x8x8x64xbf16>
    %2779 = stablehlo.broadcast_in_dim %2769, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2780 = stablehlo.multiply %2770, %2779 : tensor<1x8x8x64xbf16>
    %2781 = stablehlo.add %2778, %2780 : tensor<1x8x8x64xbf16>
    %2782 = stablehlo.concatenate %2776, %2781, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2783 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2784 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2785 = stablehlo.broadcast_in_dim %2783, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2786 = stablehlo.and %2785, %2784 : tensor<1x1x8x8xi1>
    %2787 = stablehlo.convert %2786 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2788 = sdy.sharding_constraint %2765 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2789 = sdy.sharding_constraint %2782 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2790 = sdy.sharding_constraint %2739 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2791 = sdy.sharding_constraint %2787 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2792 = stablehlo.reshape %2788 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2793 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2794 = stablehlo.multiply %2792, %2793 : tensor<1x8x8x2x128xbf16>
    %2795 = stablehlo.dot_general %2789, %2794, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2796 = stablehlo.transpose %2795, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2797 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2798 = stablehlo.compare  NE, %2791, %2797,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2799 = stablehlo.convert %2798 : tensor<1x1x8x8xi1>
    %2800 = stablehlo.broadcast_in_dim %2799, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2801 = stablehlo.reshape %2800 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2802 = call @_where_83(%2801, %2796, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2803 = stablehlo.convert %2802 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_126 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2804 = stablehlo.reduce(%2803 init: %cst_126) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2805 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2806 = stablehlo.maximum %2805, %2804 : tensor<1x8x2x8xf32>
    %2807 = stablehlo.broadcast_in_dim %2806, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2808 = stablehlo.broadcast_in_dim %2807, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2809 = stablehlo.subtract %2803, %2808 : tensor<1x8x2x8x8xf32>
    %2810 = stablehlo.exponential %2809 : tensor<1x8x2x8x8xf32>
    %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2811 = stablehlo.reduce(%2810 init: %cst_127) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2812 = stablehlo.broadcast_in_dim %2811, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2813 = stablehlo.broadcast_in_dim %2812, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2814 = stablehlo.divide %2810, %2813 : tensor<1x8x2x8x8xf32>
    %2815 = stablehlo.convert %2814 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2816 = stablehlo.dot_general %2790, %2815, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2817 = stablehlo.transpose %2816, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2818 = stablehlo.reshape %2817 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2819 = sdy.sharding_constraint %2818 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2820 = stablehlo.reshape %2819 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2821 = stablehlo.convert %arg184 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2822 = stablehlo.dot_general %2820, %2821, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2823 = stablehlo.add %2682, %2822 : tensor<1x8x1024xbf16>
    %2824 = stablehlo.convert %2823 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2825 = stablehlo.multiply %2824, %2824 : tensor<1x8x1024xf32>
    %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2826 = stablehlo.reduce(%2825 init: %cst_128) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2827 = stablehlo.broadcast_in_dim %2826, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2828 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2829 = stablehlo.divide %2827, %2828 : tensor<1x8x1xf32>
    %2830 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2831 = stablehlo.add %2829, %2830 : tensor<1x8x1xf32>
    %2832 = stablehlo.rsqrt %2831 : tensor<1x8x1xf32>
    %2833 = stablehlo.broadcast_in_dim %2832, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2834 = stablehlo.multiply %2824, %2833 : tensor<1x8x1024xf32>
    %2835 = stablehlo.convert %2834 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2836 = stablehlo.convert %arg181 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2837 = stablehlo.broadcast_in_dim %2836, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2838 = stablehlo.broadcast_in_dim %2837, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2839 = stablehlo.multiply %2838, %2835 : tensor<1x8x1024xbf16>
    %2840 = stablehlo.convert %arg179 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2841 = stablehlo.dot_general %2839, %2840, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2842 = call @silu(%2841) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2843 = stablehlo.convert %arg180 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %2844 = stablehlo.dot_general %2839, %2843, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %2845 = stablehlo.multiply %2842, %2844 : tensor<1x8x3072xbf16>
    %2846 = stablehlo.convert %arg178 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %2847 = stablehlo.dot_general %2845, %2846, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2848 = stablehlo.add %2823, %2847 : tensor<1x8x1024xbf16>
    %2849 = stablehlo.convert %2848 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2850 = stablehlo.multiply %2849, %2849 : tensor<1x8x1024xf32>
    %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2851 = stablehlo.reduce(%2850 init: %cst_129) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2852 = stablehlo.broadcast_in_dim %2851, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2853 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2854 = stablehlo.divide %2852, %2853 : tensor<1x8x1xf32>
    %2855 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2856 = stablehlo.add %2854, %2855 : tensor<1x8x1xf32>
    %2857 = stablehlo.rsqrt %2856 : tensor<1x8x1xf32>
    %2858 = stablehlo.broadcast_in_dim %2857, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %2859 = stablehlo.multiply %2849, %2858 : tensor<1x8x1024xf32>
    %2860 = stablehlo.convert %2859 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %2861 = stablehlo.convert %arg188 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %2862 = stablehlo.broadcast_in_dim %2861, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %2863 = stablehlo.broadcast_in_dim %2862, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2864 = stablehlo.multiply %2863, %2860 : tensor<1x8x1024xbf16>
    %2865 = stablehlo.convert %arg197 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %2866 = stablehlo.dot_general %2864, %2865, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %2867 = stablehlo.convert %arg194 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2868 = stablehlo.dot_general %2864, %2867, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2869 = stablehlo.convert %arg198 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %2870 = stablehlo.dot_general %2864, %2869, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2871 = stablehlo.reshape %2866 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %2872 = stablehlo.convert %2871 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %2873 = stablehlo.multiply %2872, %2872 : tensor<1x8x16x128xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2874 = stablehlo.reduce(%2873 init: %cst_130) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %2875 = stablehlo.broadcast_in_dim %2874, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %2876 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2877 = stablehlo.divide %2875, %2876 : tensor<1x8x16x1xf32>
    %2878 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %2879 = stablehlo.add %2877, %2878 : tensor<1x8x16x1xf32>
    %2880 = stablehlo.rsqrt %2879 : tensor<1x8x16x1xf32>
    %2881 = stablehlo.broadcast_in_dim %2880, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %2882 = stablehlo.multiply %2872, %2881 : tensor<1x8x16x128xf32>
    %2883 = stablehlo.convert %2882 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %2884 = stablehlo.convert %arg196 : (tensor<128xf32>) -> tensor<128xbf16>
    %2885 = stablehlo.broadcast_in_dim %2884, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2886 = stablehlo.broadcast_in_dim %2885, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2887 = stablehlo.multiply %2886, %2883 : tensor<1x8x16x128xbf16>
    %2888 = stablehlo.reshape %2868 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2889 = stablehlo.convert %2888 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %2890 = stablehlo.multiply %2889, %2889 : tensor<1x8x8x128xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2891 = stablehlo.reduce(%2890 init: %cst_131) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %2892 = stablehlo.broadcast_in_dim %2891, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %2893 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2894 = stablehlo.divide %2892, %2893 : tensor<1x8x8x1xf32>
    %2895 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %2896 = stablehlo.add %2894, %2895 : tensor<1x8x8x1xf32>
    %2897 = stablehlo.rsqrt %2896 : tensor<1x8x8x1xf32>
    %2898 = stablehlo.broadcast_in_dim %2897, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %2899 = stablehlo.multiply %2889, %2898 : tensor<1x8x8x128xf32>
    %2900 = stablehlo.convert %2899 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %2901 = stablehlo.convert %arg193 : (tensor<128xf32>) -> tensor<128xbf16>
    %2902 = stablehlo.broadcast_in_dim %2901, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %2903 = stablehlo.broadcast_in_dim %2902, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %2904 = stablehlo.multiply %2903, %2900 : tensor<1x8x8x128xbf16>
    %2905 = stablehlo.reshape %2870 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %2906 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2907 = stablehlo.compare  LT, %7, %2906,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %2908 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %2909 = stablehlo.add %7, %2908 : tensor<1x8xi32>
    %2910 = stablehlo.select %2907, %2909, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %2911 = stablehlo.broadcast_in_dim %2910, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %2912 = "stablehlo.gather"(%26, %2911) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %2913 = stablehlo.slice %2912 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2914 = stablehlo.slice %2912 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %2915 = stablehlo.broadcast_in_dim %2913, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2916 = stablehlo.convert %2915 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2917 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2918 = stablehlo.convert %2917 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2919 = stablehlo.slice %2887 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2920 = stablehlo.slice %2887 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %2921 = stablehlo.broadcast_in_dim %2916, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2922 = stablehlo.multiply %2919, %2921 : tensor<1x8x16x64xbf16>
    %2923 = stablehlo.broadcast_in_dim %2918, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2924 = stablehlo.multiply %2920, %2923 : tensor<1x8x16x64xbf16>
    %2925 = stablehlo.subtract %2922, %2924 : tensor<1x8x16x64xbf16>
    %2926 = stablehlo.broadcast_in_dim %2916, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2927 = stablehlo.multiply %2920, %2926 : tensor<1x8x16x64xbf16>
    %2928 = stablehlo.broadcast_in_dim %2918, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %2929 = stablehlo.multiply %2919, %2928 : tensor<1x8x16x64xbf16>
    %2930 = stablehlo.add %2927, %2929 : tensor<1x8x16x64xbf16>
    %2931 = stablehlo.concatenate %2925, %2930, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %2932 = stablehlo.broadcast_in_dim %2913, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2933 = stablehlo.convert %2932 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2934 = stablehlo.broadcast_in_dim %2914, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %2935 = stablehlo.convert %2934 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %2936 = stablehlo.slice %2904 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2937 = stablehlo.slice %2904 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %2938 = stablehlo.broadcast_in_dim %2933, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2939 = stablehlo.multiply %2936, %2938 : tensor<1x8x8x64xbf16>
    %2940 = stablehlo.broadcast_in_dim %2935, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2941 = stablehlo.multiply %2937, %2940 : tensor<1x8x8x64xbf16>
    %2942 = stablehlo.subtract %2939, %2941 : tensor<1x8x8x64xbf16>
    %2943 = stablehlo.broadcast_in_dim %2933, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2944 = stablehlo.multiply %2937, %2943 : tensor<1x8x8x64xbf16>
    %2945 = stablehlo.broadcast_in_dim %2935, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %2946 = stablehlo.multiply %2936, %2945 : tensor<1x8x8x64xbf16>
    %2947 = stablehlo.add %2944, %2946 : tensor<1x8x8x64xbf16>
    %2948 = stablehlo.concatenate %2942, %2947, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %2949 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %2950 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %2951 = stablehlo.broadcast_in_dim %2949, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %2952 = stablehlo.and %2951, %2950 : tensor<1x1x8x8xi1>
    %2953 = stablehlo.convert %2952 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %2954 = sdy.sharding_constraint %2931 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2955 = sdy.sharding_constraint %2948 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2956 = sdy.sharding_constraint %2905 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %2957 = sdy.sharding_constraint %2953 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %2958 = stablehlo.reshape %2954 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2959 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %2960 = stablehlo.multiply %2958, %2959 : tensor<1x8x8x2x128xbf16>
    %2961 = stablehlo.dot_general %2955, %2960, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %2962 = stablehlo.transpose %2961, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2963 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %2964 = stablehlo.compare  NE, %2957, %2963,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %2965 = stablehlo.convert %2964 : tensor<1x1x8x8xi1>
    %2966 = stablehlo.broadcast_in_dim %2965, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %2967 = stablehlo.reshape %2966 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %2968 = call @_where_83(%2967, %2962, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %2969 = stablehlo.convert %2968 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_133 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2970 = stablehlo.reduce(%2969 init: %cst_133) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2971 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %2972 = stablehlo.maximum %2971, %2970 : tensor<1x8x2x8xf32>
    %2973 = stablehlo.broadcast_in_dim %2972, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2974 = stablehlo.broadcast_in_dim %2973, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2975 = stablehlo.subtract %2969, %2974 : tensor<1x8x2x8x8xf32>
    %2976 = stablehlo.exponential %2975 : tensor<1x8x2x8x8xf32>
    %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2977 = stablehlo.reduce(%2976 init: %cst_134) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %2978 = stablehlo.broadcast_in_dim %2977, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %2979 = stablehlo.broadcast_in_dim %2978, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %2980 = stablehlo.divide %2976, %2979 : tensor<1x8x2x8x8xf32>
    %2981 = stablehlo.convert %2980 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %2982 = stablehlo.dot_general %2956, %2981, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %2983 = stablehlo.transpose %2982, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %2984 = stablehlo.reshape %2983 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %2985 = sdy.sharding_constraint %2984 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %2986 = stablehlo.reshape %2985 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %2987 = stablehlo.convert %arg195 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %2988 = stablehlo.dot_general %2986, %2987, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %2989 = stablehlo.add %2848, %2988 : tensor<1x8x1024xbf16>
    %2990 = stablehlo.convert %2989 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %2991 = stablehlo.multiply %2990, %2990 : tensor<1x8x1024xf32>
    %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2992 = stablehlo.reduce(%2991 init: %cst_135) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %2993 = stablehlo.broadcast_in_dim %2992, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %2994 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2995 = stablehlo.divide %2993, %2994 : tensor<1x8x1xf32>
    %2996 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %2997 = stablehlo.add %2995, %2996 : tensor<1x8x1xf32>
    %2998 = stablehlo.rsqrt %2997 : tensor<1x8x1xf32>
    %2999 = stablehlo.broadcast_in_dim %2998, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3000 = stablehlo.multiply %2990, %2999 : tensor<1x8x1024xf32>
    %3001 = stablehlo.convert %3000 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3002 = stablehlo.convert %arg192 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3003 = stablehlo.broadcast_in_dim %3002, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3004 = stablehlo.broadcast_in_dim %3003, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3005 = stablehlo.multiply %3004, %3001 : tensor<1x8x1024xbf16>
    %3006 = stablehlo.convert %arg190 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3007 = stablehlo.dot_general %3005, %3006, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3008 = call @silu(%3007) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3009 = stablehlo.convert %arg191 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3010 = stablehlo.dot_general %3005, %3009, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3011 = stablehlo.multiply %3008, %3010 : tensor<1x8x3072xbf16>
    %3012 = stablehlo.convert %arg189 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3013 = stablehlo.dot_general %3011, %3012, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3014 = stablehlo.add %2989, %3013 : tensor<1x8x1024xbf16>
    %3015 = stablehlo.convert %3014 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3016 = stablehlo.multiply %3015, %3015 : tensor<1x8x1024xf32>
    %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3017 = stablehlo.reduce(%3016 init: %cst_136) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3018 = stablehlo.broadcast_in_dim %3017, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3019 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3020 = stablehlo.divide %3018, %3019 : tensor<1x8x1xf32>
    %3021 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3022 = stablehlo.add %3020, %3021 : tensor<1x8x1xf32>
    %3023 = stablehlo.rsqrt %3022 : tensor<1x8x1xf32>
    %3024 = stablehlo.broadcast_in_dim %3023, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3025 = stablehlo.multiply %3015, %3024 : tensor<1x8x1024xf32>
    %3026 = stablehlo.convert %3025 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3027 = stablehlo.convert %arg199 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3028 = stablehlo.broadcast_in_dim %3027, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3029 = stablehlo.broadcast_in_dim %3028, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3030 = stablehlo.multiply %3029, %3026 : tensor<1x8x1024xbf16>
    %3031 = stablehlo.convert %arg208 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3032 = stablehlo.dot_general %3030, %3031, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3033 = stablehlo.convert %arg205 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3034 = stablehlo.dot_general %3030, %3033, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3035 = stablehlo.convert %arg209 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3036 = stablehlo.dot_general %3030, %3035, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3037 = stablehlo.reshape %3032 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3038 = stablehlo.convert %3037 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3039 = stablehlo.multiply %3038, %3038 : tensor<1x8x16x128xf32>
    %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3040 = stablehlo.reduce(%3039 init: %cst_137) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3041 = stablehlo.broadcast_in_dim %3040, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3042 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3043 = stablehlo.divide %3041, %3042 : tensor<1x8x16x1xf32>
    %3044 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3045 = stablehlo.add %3043, %3044 : tensor<1x8x16x1xf32>
    %3046 = stablehlo.rsqrt %3045 : tensor<1x8x16x1xf32>
    %3047 = stablehlo.broadcast_in_dim %3046, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3048 = stablehlo.multiply %3038, %3047 : tensor<1x8x16x128xf32>
    %3049 = stablehlo.convert %3048 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3050 = stablehlo.convert %arg207 : (tensor<128xf32>) -> tensor<128xbf16>
    %3051 = stablehlo.broadcast_in_dim %3050, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3052 = stablehlo.broadcast_in_dim %3051, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3053 = stablehlo.multiply %3052, %3049 : tensor<1x8x16x128xbf16>
    %3054 = stablehlo.reshape %3034 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3055 = stablehlo.convert %3054 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3056 = stablehlo.multiply %3055, %3055 : tensor<1x8x8x128xf32>
    %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3057 = stablehlo.reduce(%3056 init: %cst_138) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3058 = stablehlo.broadcast_in_dim %3057, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3059 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3060 = stablehlo.divide %3058, %3059 : tensor<1x8x8x1xf32>
    %3061 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3062 = stablehlo.add %3060, %3061 : tensor<1x8x8x1xf32>
    %3063 = stablehlo.rsqrt %3062 : tensor<1x8x8x1xf32>
    %3064 = stablehlo.broadcast_in_dim %3063, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3065 = stablehlo.multiply %3055, %3064 : tensor<1x8x8x128xf32>
    %3066 = stablehlo.convert %3065 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3067 = stablehlo.convert %arg204 : (tensor<128xf32>) -> tensor<128xbf16>
    %3068 = stablehlo.broadcast_in_dim %3067, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3069 = stablehlo.broadcast_in_dim %3068, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3070 = stablehlo.multiply %3069, %3066 : tensor<1x8x8x128xbf16>
    %3071 = stablehlo.reshape %3036 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3072 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3073 = stablehlo.compare  LT, %7, %3072,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3074 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3075 = stablehlo.add %7, %3074 : tensor<1x8xi32>
    %3076 = stablehlo.select %3073, %3075, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3077 = stablehlo.broadcast_in_dim %3076, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3078 = "stablehlo.gather"(%26, %3077) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3079 = stablehlo.slice %3078 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3080 = stablehlo.slice %3078 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3081 = stablehlo.broadcast_in_dim %3079, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3082 = stablehlo.convert %3081 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3083 = stablehlo.broadcast_in_dim %3080, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3084 = stablehlo.convert %3083 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3085 = stablehlo.slice %3053 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3086 = stablehlo.slice %3053 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3087 = stablehlo.broadcast_in_dim %3082, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3088 = stablehlo.multiply %3085, %3087 : tensor<1x8x16x64xbf16>
    %3089 = stablehlo.broadcast_in_dim %3084, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3090 = stablehlo.multiply %3086, %3089 : tensor<1x8x16x64xbf16>
    %3091 = stablehlo.subtract %3088, %3090 : tensor<1x8x16x64xbf16>
    %3092 = stablehlo.broadcast_in_dim %3082, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3093 = stablehlo.multiply %3086, %3092 : tensor<1x8x16x64xbf16>
    %3094 = stablehlo.broadcast_in_dim %3084, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3095 = stablehlo.multiply %3085, %3094 : tensor<1x8x16x64xbf16>
    %3096 = stablehlo.add %3093, %3095 : tensor<1x8x16x64xbf16>
    %3097 = stablehlo.concatenate %3091, %3096, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3098 = stablehlo.broadcast_in_dim %3079, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3099 = stablehlo.convert %3098 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3100 = stablehlo.broadcast_in_dim %3080, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3101 = stablehlo.convert %3100 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3102 = stablehlo.slice %3070 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3103 = stablehlo.slice %3070 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3104 = stablehlo.broadcast_in_dim %3099, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3105 = stablehlo.multiply %3102, %3104 : tensor<1x8x8x64xbf16>
    %3106 = stablehlo.broadcast_in_dim %3101, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3107 = stablehlo.multiply %3103, %3106 : tensor<1x8x8x64xbf16>
    %3108 = stablehlo.subtract %3105, %3107 : tensor<1x8x8x64xbf16>
    %3109 = stablehlo.broadcast_in_dim %3099, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3110 = stablehlo.multiply %3103, %3109 : tensor<1x8x8x64xbf16>
    %3111 = stablehlo.broadcast_in_dim %3101, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3112 = stablehlo.multiply %3102, %3111 : tensor<1x8x8x64xbf16>
    %3113 = stablehlo.add %3110, %3112 : tensor<1x8x8x64xbf16>
    %3114 = stablehlo.concatenate %3108, %3113, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3115 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3116 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3117 = stablehlo.broadcast_in_dim %3115, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3118 = stablehlo.and %3117, %3116 : tensor<1x1x8x8xi1>
    %3119 = stablehlo.convert %3118 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3120 = sdy.sharding_constraint %3097 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3121 = sdy.sharding_constraint %3114 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3122 = sdy.sharding_constraint %3071 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3123 = sdy.sharding_constraint %3119 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3124 = stablehlo.reshape %3120 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3125 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3126 = stablehlo.multiply %3124, %3125 : tensor<1x8x8x2x128xbf16>
    %3127 = stablehlo.dot_general %3121, %3126, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3128 = stablehlo.transpose %3127, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3129 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3130 = stablehlo.compare  NE, %3123, %3129,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3131 = stablehlo.convert %3130 : tensor<1x1x8x8xi1>
    %3132 = stablehlo.broadcast_in_dim %3131, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3133 = stablehlo.reshape %3132 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3134 = call @_where_83(%3133, %3128, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3135 = stablehlo.convert %3134 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_140 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3136 = stablehlo.reduce(%3135 init: %cst_140) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3137 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3138 = stablehlo.maximum %3137, %3136 : tensor<1x8x2x8xf32>
    %3139 = stablehlo.broadcast_in_dim %3138, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3140 = stablehlo.broadcast_in_dim %3139, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3141 = stablehlo.subtract %3135, %3140 : tensor<1x8x2x8x8xf32>
    %3142 = stablehlo.exponential %3141 : tensor<1x8x2x8x8xf32>
    %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3143 = stablehlo.reduce(%3142 init: %cst_141) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3144 = stablehlo.broadcast_in_dim %3143, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3145 = stablehlo.broadcast_in_dim %3144, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3146 = stablehlo.divide %3142, %3145 : tensor<1x8x2x8x8xf32>
    %3147 = stablehlo.convert %3146 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3148 = stablehlo.dot_general %3122, %3147, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3149 = stablehlo.transpose %3148, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3150 = stablehlo.reshape %3149 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3151 = sdy.sharding_constraint %3150 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3152 = stablehlo.reshape %3151 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3153 = stablehlo.convert %arg206 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3154 = stablehlo.dot_general %3152, %3153, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3155 = stablehlo.add %3014, %3154 : tensor<1x8x1024xbf16>
    %3156 = stablehlo.convert %3155 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3157 = stablehlo.multiply %3156, %3156 : tensor<1x8x1024xf32>
    %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3158 = stablehlo.reduce(%3157 init: %cst_142) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3159 = stablehlo.broadcast_in_dim %3158, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3160 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3161 = stablehlo.divide %3159, %3160 : tensor<1x8x1xf32>
    %3162 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3163 = stablehlo.add %3161, %3162 : tensor<1x8x1xf32>
    %3164 = stablehlo.rsqrt %3163 : tensor<1x8x1xf32>
    %3165 = stablehlo.broadcast_in_dim %3164, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3166 = stablehlo.multiply %3156, %3165 : tensor<1x8x1024xf32>
    %3167 = stablehlo.convert %3166 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3168 = stablehlo.convert %arg203 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3169 = stablehlo.broadcast_in_dim %3168, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3170 = stablehlo.broadcast_in_dim %3169, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3171 = stablehlo.multiply %3170, %3167 : tensor<1x8x1024xbf16>
    %3172 = stablehlo.convert %arg201 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3173 = stablehlo.dot_general %3171, %3172, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3174 = call @silu(%3173) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3175 = stablehlo.convert %arg202 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3176 = stablehlo.dot_general %3171, %3175, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3177 = stablehlo.multiply %3174, %3176 : tensor<1x8x3072xbf16>
    %3178 = stablehlo.convert %arg200 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3179 = stablehlo.dot_general %3177, %3178, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3180 = stablehlo.add %3155, %3179 : tensor<1x8x1024xbf16>
    %3181 = stablehlo.convert %3180 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3182 = stablehlo.multiply %3181, %3181 : tensor<1x8x1024xf32>
    %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3183 = stablehlo.reduce(%3182 init: %cst_143) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3184 = stablehlo.broadcast_in_dim %3183, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3185 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3186 = stablehlo.divide %3184, %3185 : tensor<1x8x1xf32>
    %3187 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3188 = stablehlo.add %3186, %3187 : tensor<1x8x1xf32>
    %3189 = stablehlo.rsqrt %3188 : tensor<1x8x1xf32>
    %3190 = stablehlo.broadcast_in_dim %3189, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3191 = stablehlo.multiply %3181, %3190 : tensor<1x8x1024xf32>
    %3192 = stablehlo.convert %3191 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3193 = stablehlo.convert %arg210 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3194 = stablehlo.broadcast_in_dim %3193, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3195 = stablehlo.broadcast_in_dim %3194, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3196 = stablehlo.multiply %3195, %3192 : tensor<1x8x1024xbf16>
    %3197 = stablehlo.convert %arg219 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3198 = stablehlo.dot_general %3196, %3197, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3199 = stablehlo.convert %arg216 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3200 = stablehlo.dot_general %3196, %3199, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3201 = stablehlo.convert %arg220 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3202 = stablehlo.dot_general %3196, %3201, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3203 = stablehlo.reshape %3198 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3204 = stablehlo.convert %3203 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3205 = stablehlo.multiply %3204, %3204 : tensor<1x8x16x128xf32>
    %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3206 = stablehlo.reduce(%3205 init: %cst_144) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3207 = stablehlo.broadcast_in_dim %3206, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3208 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3209 = stablehlo.divide %3207, %3208 : tensor<1x8x16x1xf32>
    %3210 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3211 = stablehlo.add %3209, %3210 : tensor<1x8x16x1xf32>
    %3212 = stablehlo.rsqrt %3211 : tensor<1x8x16x1xf32>
    %3213 = stablehlo.broadcast_in_dim %3212, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3214 = stablehlo.multiply %3204, %3213 : tensor<1x8x16x128xf32>
    %3215 = stablehlo.convert %3214 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3216 = stablehlo.convert %arg218 : (tensor<128xf32>) -> tensor<128xbf16>
    %3217 = stablehlo.broadcast_in_dim %3216, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3218 = stablehlo.broadcast_in_dim %3217, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3219 = stablehlo.multiply %3218, %3215 : tensor<1x8x16x128xbf16>
    %3220 = stablehlo.reshape %3200 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3221 = stablehlo.convert %3220 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3222 = stablehlo.multiply %3221, %3221 : tensor<1x8x8x128xf32>
    %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3223 = stablehlo.reduce(%3222 init: %cst_145) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3224 = stablehlo.broadcast_in_dim %3223, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3225 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3226 = stablehlo.divide %3224, %3225 : tensor<1x8x8x1xf32>
    %3227 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3228 = stablehlo.add %3226, %3227 : tensor<1x8x8x1xf32>
    %3229 = stablehlo.rsqrt %3228 : tensor<1x8x8x1xf32>
    %3230 = stablehlo.broadcast_in_dim %3229, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3231 = stablehlo.multiply %3221, %3230 : tensor<1x8x8x128xf32>
    %3232 = stablehlo.convert %3231 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3233 = stablehlo.convert %arg215 : (tensor<128xf32>) -> tensor<128xbf16>
    %3234 = stablehlo.broadcast_in_dim %3233, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3235 = stablehlo.broadcast_in_dim %3234, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3236 = stablehlo.multiply %3235, %3232 : tensor<1x8x8x128xbf16>
    %3237 = stablehlo.reshape %3202 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3238 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3239 = stablehlo.compare  LT, %7, %3238,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3240 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3241 = stablehlo.add %7, %3240 : tensor<1x8xi32>
    %3242 = stablehlo.select %3239, %3241, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3243 = stablehlo.broadcast_in_dim %3242, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3244 = "stablehlo.gather"(%26, %3243) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3245 = stablehlo.slice %3244 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3246 = stablehlo.slice %3244 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3247 = stablehlo.broadcast_in_dim %3245, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3248 = stablehlo.convert %3247 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3249 = stablehlo.broadcast_in_dim %3246, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3250 = stablehlo.convert %3249 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3251 = stablehlo.slice %3219 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3252 = stablehlo.slice %3219 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3253 = stablehlo.broadcast_in_dim %3248, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3254 = stablehlo.multiply %3251, %3253 : tensor<1x8x16x64xbf16>
    %3255 = stablehlo.broadcast_in_dim %3250, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3256 = stablehlo.multiply %3252, %3255 : tensor<1x8x16x64xbf16>
    %3257 = stablehlo.subtract %3254, %3256 : tensor<1x8x16x64xbf16>
    %3258 = stablehlo.broadcast_in_dim %3248, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3259 = stablehlo.multiply %3252, %3258 : tensor<1x8x16x64xbf16>
    %3260 = stablehlo.broadcast_in_dim %3250, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3261 = stablehlo.multiply %3251, %3260 : tensor<1x8x16x64xbf16>
    %3262 = stablehlo.add %3259, %3261 : tensor<1x8x16x64xbf16>
    %3263 = stablehlo.concatenate %3257, %3262, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3264 = stablehlo.broadcast_in_dim %3245, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3265 = stablehlo.convert %3264 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3266 = stablehlo.broadcast_in_dim %3246, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3267 = stablehlo.convert %3266 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3268 = stablehlo.slice %3236 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3269 = stablehlo.slice %3236 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3270 = stablehlo.broadcast_in_dim %3265, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3271 = stablehlo.multiply %3268, %3270 : tensor<1x8x8x64xbf16>
    %3272 = stablehlo.broadcast_in_dim %3267, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3273 = stablehlo.multiply %3269, %3272 : tensor<1x8x8x64xbf16>
    %3274 = stablehlo.subtract %3271, %3273 : tensor<1x8x8x64xbf16>
    %3275 = stablehlo.broadcast_in_dim %3265, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3276 = stablehlo.multiply %3269, %3275 : tensor<1x8x8x64xbf16>
    %3277 = stablehlo.broadcast_in_dim %3267, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3278 = stablehlo.multiply %3268, %3277 : tensor<1x8x8x64xbf16>
    %3279 = stablehlo.add %3276, %3278 : tensor<1x8x8x64xbf16>
    %3280 = stablehlo.concatenate %3274, %3279, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3281 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3282 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3283 = stablehlo.broadcast_in_dim %3281, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3284 = stablehlo.and %3283, %3282 : tensor<1x1x8x8xi1>
    %3285 = stablehlo.convert %3284 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3286 = sdy.sharding_constraint %3263 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3287 = sdy.sharding_constraint %3280 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3288 = sdy.sharding_constraint %3237 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3289 = sdy.sharding_constraint %3285 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3290 = stablehlo.reshape %3286 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3291 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3292 = stablehlo.multiply %3290, %3291 : tensor<1x8x8x2x128xbf16>
    %3293 = stablehlo.dot_general %3287, %3292, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3294 = stablehlo.transpose %3293, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3295 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3296 = stablehlo.compare  NE, %3289, %3295,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3297 = stablehlo.convert %3296 : tensor<1x1x8x8xi1>
    %3298 = stablehlo.broadcast_in_dim %3297, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3299 = stablehlo.reshape %3298 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3300 = call @_where_83(%3299, %3294, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3301 = stablehlo.convert %3300 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_147 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3302 = stablehlo.reduce(%3301 init: %cst_147) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3303 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3304 = stablehlo.maximum %3303, %3302 : tensor<1x8x2x8xf32>
    %3305 = stablehlo.broadcast_in_dim %3304, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3306 = stablehlo.broadcast_in_dim %3305, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3307 = stablehlo.subtract %3301, %3306 : tensor<1x8x2x8x8xf32>
    %3308 = stablehlo.exponential %3307 : tensor<1x8x2x8x8xf32>
    %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3309 = stablehlo.reduce(%3308 init: %cst_148) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3310 = stablehlo.broadcast_in_dim %3309, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3311 = stablehlo.broadcast_in_dim %3310, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3312 = stablehlo.divide %3308, %3311 : tensor<1x8x2x8x8xf32>
    %3313 = stablehlo.convert %3312 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3314 = stablehlo.dot_general %3288, %3313, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3315 = stablehlo.transpose %3314, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3316 = stablehlo.reshape %3315 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3317 = sdy.sharding_constraint %3316 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3318 = stablehlo.reshape %3317 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3319 = stablehlo.convert %arg217 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3320 = stablehlo.dot_general %3318, %3319, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3321 = stablehlo.add %3180, %3320 : tensor<1x8x1024xbf16>
    %3322 = stablehlo.convert %3321 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3323 = stablehlo.multiply %3322, %3322 : tensor<1x8x1024xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3324 = stablehlo.reduce(%3323 init: %cst_149) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3325 = stablehlo.broadcast_in_dim %3324, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3326 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3327 = stablehlo.divide %3325, %3326 : tensor<1x8x1xf32>
    %3328 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3329 = stablehlo.add %3327, %3328 : tensor<1x8x1xf32>
    %3330 = stablehlo.rsqrt %3329 : tensor<1x8x1xf32>
    %3331 = stablehlo.broadcast_in_dim %3330, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3332 = stablehlo.multiply %3322, %3331 : tensor<1x8x1024xf32>
    %3333 = stablehlo.convert %3332 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3334 = stablehlo.convert %arg214 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3335 = stablehlo.broadcast_in_dim %3334, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3336 = stablehlo.broadcast_in_dim %3335, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3337 = stablehlo.multiply %3336, %3333 : tensor<1x8x1024xbf16>
    %3338 = stablehlo.convert %arg212 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3339 = stablehlo.dot_general %3337, %3338, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3340 = call @silu(%3339) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3341 = stablehlo.convert %arg213 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3342 = stablehlo.dot_general %3337, %3341, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3343 = stablehlo.multiply %3340, %3342 : tensor<1x8x3072xbf16>
    %3344 = stablehlo.convert %arg211 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3345 = stablehlo.dot_general %3343, %3344, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3346 = stablehlo.add %3321, %3345 : tensor<1x8x1024xbf16>
    %3347 = stablehlo.convert %3346 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3348 = stablehlo.multiply %3347, %3347 : tensor<1x8x1024xf32>
    %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3349 = stablehlo.reduce(%3348 init: %cst_150) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3350 = stablehlo.broadcast_in_dim %3349, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3351 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3352 = stablehlo.divide %3350, %3351 : tensor<1x8x1xf32>
    %3353 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3354 = stablehlo.add %3352, %3353 : tensor<1x8x1xf32>
    %3355 = stablehlo.rsqrt %3354 : tensor<1x8x1xf32>
    %3356 = stablehlo.broadcast_in_dim %3355, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3357 = stablehlo.multiply %3347, %3356 : tensor<1x8x1024xf32>
    %3358 = stablehlo.convert %3357 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3359 = stablehlo.convert %arg221 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3360 = stablehlo.broadcast_in_dim %3359, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3361 = stablehlo.broadcast_in_dim %3360, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3362 = stablehlo.multiply %3361, %3358 : tensor<1x8x1024xbf16>
    %3363 = stablehlo.convert %arg230 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3364 = stablehlo.dot_general %3362, %3363, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3365 = stablehlo.convert %arg227 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3366 = stablehlo.dot_general %3362, %3365, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3367 = stablehlo.convert %arg231 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3368 = stablehlo.dot_general %3362, %3367, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3369 = stablehlo.reshape %3364 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3370 = stablehlo.convert %3369 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3371 = stablehlo.multiply %3370, %3370 : tensor<1x8x16x128xf32>
    %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3372 = stablehlo.reduce(%3371 init: %cst_151) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3373 = stablehlo.broadcast_in_dim %3372, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3374 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3375 = stablehlo.divide %3373, %3374 : tensor<1x8x16x1xf32>
    %3376 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3377 = stablehlo.add %3375, %3376 : tensor<1x8x16x1xf32>
    %3378 = stablehlo.rsqrt %3377 : tensor<1x8x16x1xf32>
    %3379 = stablehlo.broadcast_in_dim %3378, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3380 = stablehlo.multiply %3370, %3379 : tensor<1x8x16x128xf32>
    %3381 = stablehlo.convert %3380 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3382 = stablehlo.convert %arg229 : (tensor<128xf32>) -> tensor<128xbf16>
    %3383 = stablehlo.broadcast_in_dim %3382, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3384 = stablehlo.broadcast_in_dim %3383, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3385 = stablehlo.multiply %3384, %3381 : tensor<1x8x16x128xbf16>
    %3386 = stablehlo.reshape %3366 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3387 = stablehlo.convert %3386 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3388 = stablehlo.multiply %3387, %3387 : tensor<1x8x8x128xf32>
    %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3389 = stablehlo.reduce(%3388 init: %cst_152) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3390 = stablehlo.broadcast_in_dim %3389, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3391 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3392 = stablehlo.divide %3390, %3391 : tensor<1x8x8x1xf32>
    %3393 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3394 = stablehlo.add %3392, %3393 : tensor<1x8x8x1xf32>
    %3395 = stablehlo.rsqrt %3394 : tensor<1x8x8x1xf32>
    %3396 = stablehlo.broadcast_in_dim %3395, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3397 = stablehlo.multiply %3387, %3396 : tensor<1x8x8x128xf32>
    %3398 = stablehlo.convert %3397 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3399 = stablehlo.convert %arg226 : (tensor<128xf32>) -> tensor<128xbf16>
    %3400 = stablehlo.broadcast_in_dim %3399, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3401 = stablehlo.broadcast_in_dim %3400, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3402 = stablehlo.multiply %3401, %3398 : tensor<1x8x8x128xbf16>
    %3403 = stablehlo.reshape %3368 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3404 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3405 = stablehlo.compare  LT, %7, %3404,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3406 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3407 = stablehlo.add %7, %3406 : tensor<1x8xi32>
    %3408 = stablehlo.select %3405, %3407, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3409 = stablehlo.broadcast_in_dim %3408, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3410 = "stablehlo.gather"(%26, %3409) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3411 = stablehlo.slice %3410 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3412 = stablehlo.slice %3410 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3413 = stablehlo.broadcast_in_dim %3411, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3414 = stablehlo.convert %3413 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3415 = stablehlo.broadcast_in_dim %3412, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3416 = stablehlo.convert %3415 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3417 = stablehlo.slice %3385 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3418 = stablehlo.slice %3385 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3419 = stablehlo.broadcast_in_dim %3414, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3420 = stablehlo.multiply %3417, %3419 : tensor<1x8x16x64xbf16>
    %3421 = stablehlo.broadcast_in_dim %3416, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3422 = stablehlo.multiply %3418, %3421 : tensor<1x8x16x64xbf16>
    %3423 = stablehlo.subtract %3420, %3422 : tensor<1x8x16x64xbf16>
    %3424 = stablehlo.broadcast_in_dim %3414, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3425 = stablehlo.multiply %3418, %3424 : tensor<1x8x16x64xbf16>
    %3426 = stablehlo.broadcast_in_dim %3416, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3427 = stablehlo.multiply %3417, %3426 : tensor<1x8x16x64xbf16>
    %3428 = stablehlo.add %3425, %3427 : tensor<1x8x16x64xbf16>
    %3429 = stablehlo.concatenate %3423, %3428, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3430 = stablehlo.broadcast_in_dim %3411, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3431 = stablehlo.convert %3430 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3432 = stablehlo.broadcast_in_dim %3412, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3433 = stablehlo.convert %3432 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3434 = stablehlo.slice %3402 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3435 = stablehlo.slice %3402 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3436 = stablehlo.broadcast_in_dim %3431, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3437 = stablehlo.multiply %3434, %3436 : tensor<1x8x8x64xbf16>
    %3438 = stablehlo.broadcast_in_dim %3433, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3439 = stablehlo.multiply %3435, %3438 : tensor<1x8x8x64xbf16>
    %3440 = stablehlo.subtract %3437, %3439 : tensor<1x8x8x64xbf16>
    %3441 = stablehlo.broadcast_in_dim %3431, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3442 = stablehlo.multiply %3435, %3441 : tensor<1x8x8x64xbf16>
    %3443 = stablehlo.broadcast_in_dim %3433, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3444 = stablehlo.multiply %3434, %3443 : tensor<1x8x8x64xbf16>
    %3445 = stablehlo.add %3442, %3444 : tensor<1x8x8x64xbf16>
    %3446 = stablehlo.concatenate %3440, %3445, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3447 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3448 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3449 = stablehlo.broadcast_in_dim %3447, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3450 = stablehlo.and %3449, %3448 : tensor<1x1x8x8xi1>
    %3451 = stablehlo.convert %3450 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3452 = sdy.sharding_constraint %3429 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3453 = sdy.sharding_constraint %3446 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3454 = sdy.sharding_constraint %3403 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3455 = sdy.sharding_constraint %3451 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3456 = stablehlo.reshape %3452 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3457 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3458 = stablehlo.multiply %3456, %3457 : tensor<1x8x8x2x128xbf16>
    %3459 = stablehlo.dot_general %3453, %3458, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3460 = stablehlo.transpose %3459, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3461 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3462 = stablehlo.compare  NE, %3455, %3461,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3463 = stablehlo.convert %3462 : tensor<1x1x8x8xi1>
    %3464 = stablehlo.broadcast_in_dim %3463, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3465 = stablehlo.reshape %3464 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3466 = call @_where_83(%3465, %3460, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3467 = stablehlo.convert %3466 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_154 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3468 = stablehlo.reduce(%3467 init: %cst_154) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3469 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3470 = stablehlo.maximum %3469, %3468 : tensor<1x8x2x8xf32>
    %3471 = stablehlo.broadcast_in_dim %3470, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3472 = stablehlo.broadcast_in_dim %3471, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3473 = stablehlo.subtract %3467, %3472 : tensor<1x8x2x8x8xf32>
    %3474 = stablehlo.exponential %3473 : tensor<1x8x2x8x8xf32>
    %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3475 = stablehlo.reduce(%3474 init: %cst_155) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3476 = stablehlo.broadcast_in_dim %3475, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3477 = stablehlo.broadcast_in_dim %3476, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3478 = stablehlo.divide %3474, %3477 : tensor<1x8x2x8x8xf32>
    %3479 = stablehlo.convert %3478 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3480 = stablehlo.dot_general %3454, %3479, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3481 = stablehlo.transpose %3480, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3482 = stablehlo.reshape %3481 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3483 = sdy.sharding_constraint %3482 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3484 = stablehlo.reshape %3483 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3485 = stablehlo.convert %arg228 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3486 = stablehlo.dot_general %3484, %3485, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3487 = stablehlo.add %3346, %3486 : tensor<1x8x1024xbf16>
    %3488 = stablehlo.convert %3487 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3489 = stablehlo.multiply %3488, %3488 : tensor<1x8x1024xf32>
    %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3490 = stablehlo.reduce(%3489 init: %cst_156) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3491 = stablehlo.broadcast_in_dim %3490, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3492 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3493 = stablehlo.divide %3491, %3492 : tensor<1x8x1xf32>
    %3494 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3495 = stablehlo.add %3493, %3494 : tensor<1x8x1xf32>
    %3496 = stablehlo.rsqrt %3495 : tensor<1x8x1xf32>
    %3497 = stablehlo.broadcast_in_dim %3496, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3498 = stablehlo.multiply %3488, %3497 : tensor<1x8x1024xf32>
    %3499 = stablehlo.convert %3498 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3500 = stablehlo.convert %arg225 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3501 = stablehlo.broadcast_in_dim %3500, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3502 = stablehlo.broadcast_in_dim %3501, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3503 = stablehlo.multiply %3502, %3499 : tensor<1x8x1024xbf16>
    %3504 = stablehlo.convert %arg223 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3505 = stablehlo.dot_general %3503, %3504, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3506 = call @silu(%3505) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3507 = stablehlo.convert %arg224 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3508 = stablehlo.dot_general %3503, %3507, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3509 = stablehlo.multiply %3506, %3508 : tensor<1x8x3072xbf16>
    %3510 = stablehlo.convert %arg222 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3511 = stablehlo.dot_general %3509, %3510, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3512 = stablehlo.add %3487, %3511 : tensor<1x8x1024xbf16>
    %3513 = stablehlo.convert %3512 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3514 = stablehlo.multiply %3513, %3513 : tensor<1x8x1024xf32>
    %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3515 = stablehlo.reduce(%3514 init: %cst_157) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3516 = stablehlo.broadcast_in_dim %3515, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3517 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3518 = stablehlo.divide %3516, %3517 : tensor<1x8x1xf32>
    %3519 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3520 = stablehlo.add %3518, %3519 : tensor<1x8x1xf32>
    %3521 = stablehlo.rsqrt %3520 : tensor<1x8x1xf32>
    %3522 = stablehlo.broadcast_in_dim %3521, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3523 = stablehlo.multiply %3513, %3522 : tensor<1x8x1024xf32>
    %3524 = stablehlo.convert %3523 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3525 = stablehlo.convert %arg232 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3526 = stablehlo.broadcast_in_dim %3525, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3527 = stablehlo.broadcast_in_dim %3526, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3528 = stablehlo.multiply %3527, %3524 : tensor<1x8x1024xbf16>
    %3529 = stablehlo.convert %arg241 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3530 = stablehlo.dot_general %3528, %3529, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3531 = stablehlo.convert %arg238 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3532 = stablehlo.dot_general %3528, %3531, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3533 = stablehlo.convert %arg242 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3534 = stablehlo.dot_general %3528, %3533, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3535 = stablehlo.reshape %3530 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3536 = stablehlo.convert %3535 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3537 = stablehlo.multiply %3536, %3536 : tensor<1x8x16x128xf32>
    %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3538 = stablehlo.reduce(%3537 init: %cst_158) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3539 = stablehlo.broadcast_in_dim %3538, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3540 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3541 = stablehlo.divide %3539, %3540 : tensor<1x8x16x1xf32>
    %3542 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3543 = stablehlo.add %3541, %3542 : tensor<1x8x16x1xf32>
    %3544 = stablehlo.rsqrt %3543 : tensor<1x8x16x1xf32>
    %3545 = stablehlo.broadcast_in_dim %3544, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3546 = stablehlo.multiply %3536, %3545 : tensor<1x8x16x128xf32>
    %3547 = stablehlo.convert %3546 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3548 = stablehlo.convert %arg240 : (tensor<128xf32>) -> tensor<128xbf16>
    %3549 = stablehlo.broadcast_in_dim %3548, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3550 = stablehlo.broadcast_in_dim %3549, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3551 = stablehlo.multiply %3550, %3547 : tensor<1x8x16x128xbf16>
    %3552 = stablehlo.reshape %3532 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3553 = stablehlo.convert %3552 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3554 = stablehlo.multiply %3553, %3553 : tensor<1x8x8x128xf32>
    %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3555 = stablehlo.reduce(%3554 init: %cst_159) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3556 = stablehlo.broadcast_in_dim %3555, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3557 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3558 = stablehlo.divide %3556, %3557 : tensor<1x8x8x1xf32>
    %3559 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3560 = stablehlo.add %3558, %3559 : tensor<1x8x8x1xf32>
    %3561 = stablehlo.rsqrt %3560 : tensor<1x8x8x1xf32>
    %3562 = stablehlo.broadcast_in_dim %3561, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3563 = stablehlo.multiply %3553, %3562 : tensor<1x8x8x128xf32>
    %3564 = stablehlo.convert %3563 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3565 = stablehlo.convert %arg237 : (tensor<128xf32>) -> tensor<128xbf16>
    %3566 = stablehlo.broadcast_in_dim %3565, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3567 = stablehlo.broadcast_in_dim %3566, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3568 = stablehlo.multiply %3567, %3564 : tensor<1x8x8x128xbf16>
    %3569 = stablehlo.reshape %3534 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3570 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3571 = stablehlo.compare  LT, %7, %3570,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3572 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3573 = stablehlo.add %7, %3572 : tensor<1x8xi32>
    %3574 = stablehlo.select %3571, %3573, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3575 = stablehlo.broadcast_in_dim %3574, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3576 = "stablehlo.gather"(%26, %3575) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3577 = stablehlo.slice %3576 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3578 = stablehlo.slice %3576 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3579 = stablehlo.broadcast_in_dim %3577, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3580 = stablehlo.convert %3579 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3581 = stablehlo.broadcast_in_dim %3578, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3582 = stablehlo.convert %3581 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3583 = stablehlo.slice %3551 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3584 = stablehlo.slice %3551 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3585 = stablehlo.broadcast_in_dim %3580, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3586 = stablehlo.multiply %3583, %3585 : tensor<1x8x16x64xbf16>
    %3587 = stablehlo.broadcast_in_dim %3582, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3588 = stablehlo.multiply %3584, %3587 : tensor<1x8x16x64xbf16>
    %3589 = stablehlo.subtract %3586, %3588 : tensor<1x8x16x64xbf16>
    %3590 = stablehlo.broadcast_in_dim %3580, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3591 = stablehlo.multiply %3584, %3590 : tensor<1x8x16x64xbf16>
    %3592 = stablehlo.broadcast_in_dim %3582, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3593 = stablehlo.multiply %3583, %3592 : tensor<1x8x16x64xbf16>
    %3594 = stablehlo.add %3591, %3593 : tensor<1x8x16x64xbf16>
    %3595 = stablehlo.concatenate %3589, %3594, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3596 = stablehlo.broadcast_in_dim %3577, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3597 = stablehlo.convert %3596 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3598 = stablehlo.broadcast_in_dim %3578, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3599 = stablehlo.convert %3598 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3600 = stablehlo.slice %3568 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3601 = stablehlo.slice %3568 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3602 = stablehlo.broadcast_in_dim %3597, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3603 = stablehlo.multiply %3600, %3602 : tensor<1x8x8x64xbf16>
    %3604 = stablehlo.broadcast_in_dim %3599, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3605 = stablehlo.multiply %3601, %3604 : tensor<1x8x8x64xbf16>
    %3606 = stablehlo.subtract %3603, %3605 : tensor<1x8x8x64xbf16>
    %3607 = stablehlo.broadcast_in_dim %3597, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3608 = stablehlo.multiply %3601, %3607 : tensor<1x8x8x64xbf16>
    %3609 = stablehlo.broadcast_in_dim %3599, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3610 = stablehlo.multiply %3600, %3609 : tensor<1x8x8x64xbf16>
    %3611 = stablehlo.add %3608, %3610 : tensor<1x8x8x64xbf16>
    %3612 = stablehlo.concatenate %3606, %3611, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3613 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3614 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3615 = stablehlo.broadcast_in_dim %3613, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3616 = stablehlo.and %3615, %3614 : tensor<1x1x8x8xi1>
    %3617 = stablehlo.convert %3616 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3618 = sdy.sharding_constraint %3595 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3619 = sdy.sharding_constraint %3612 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3620 = sdy.sharding_constraint %3569 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3621 = sdy.sharding_constraint %3617 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3622 = stablehlo.reshape %3618 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3623 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3624 = stablehlo.multiply %3622, %3623 : tensor<1x8x8x2x128xbf16>
    %3625 = stablehlo.dot_general %3619, %3624, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3626 = stablehlo.transpose %3625, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3627 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3628 = stablehlo.compare  NE, %3621, %3627,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3629 = stablehlo.convert %3628 : tensor<1x1x8x8xi1>
    %3630 = stablehlo.broadcast_in_dim %3629, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3631 = stablehlo.reshape %3630 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3632 = call @_where_83(%3631, %3626, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3633 = stablehlo.convert %3632 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_161 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3634 = stablehlo.reduce(%3633 init: %cst_161) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3635 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3636 = stablehlo.maximum %3635, %3634 : tensor<1x8x2x8xf32>
    %3637 = stablehlo.broadcast_in_dim %3636, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3638 = stablehlo.broadcast_in_dim %3637, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3639 = stablehlo.subtract %3633, %3638 : tensor<1x8x2x8x8xf32>
    %3640 = stablehlo.exponential %3639 : tensor<1x8x2x8x8xf32>
    %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3641 = stablehlo.reduce(%3640 init: %cst_162) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3642 = stablehlo.broadcast_in_dim %3641, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3643 = stablehlo.broadcast_in_dim %3642, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3644 = stablehlo.divide %3640, %3643 : tensor<1x8x2x8x8xf32>
    %3645 = stablehlo.convert %3644 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3646 = stablehlo.dot_general %3620, %3645, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3647 = stablehlo.transpose %3646, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3648 = stablehlo.reshape %3647 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3649 = sdy.sharding_constraint %3648 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3650 = stablehlo.reshape %3649 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3651 = stablehlo.convert %arg239 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3652 = stablehlo.dot_general %3650, %3651, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3653 = stablehlo.add %3512, %3652 : tensor<1x8x1024xbf16>
    %3654 = stablehlo.convert %3653 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3655 = stablehlo.multiply %3654, %3654 : tensor<1x8x1024xf32>
    %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3656 = stablehlo.reduce(%3655 init: %cst_163) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3657 = stablehlo.broadcast_in_dim %3656, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3658 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3659 = stablehlo.divide %3657, %3658 : tensor<1x8x1xf32>
    %3660 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3661 = stablehlo.add %3659, %3660 : tensor<1x8x1xf32>
    %3662 = stablehlo.rsqrt %3661 : tensor<1x8x1xf32>
    %3663 = stablehlo.broadcast_in_dim %3662, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3664 = stablehlo.multiply %3654, %3663 : tensor<1x8x1024xf32>
    %3665 = stablehlo.convert %3664 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3666 = stablehlo.convert %arg236 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3667 = stablehlo.broadcast_in_dim %3666, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3668 = stablehlo.broadcast_in_dim %3667, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3669 = stablehlo.multiply %3668, %3665 : tensor<1x8x1024xbf16>
    %3670 = stablehlo.convert %arg234 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3671 = stablehlo.dot_general %3669, %3670, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3672 = call @silu(%3671) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3673 = stablehlo.convert %arg235 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3674 = stablehlo.dot_general %3669, %3673, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3675 = stablehlo.multiply %3672, %3674 : tensor<1x8x3072xbf16>
    %3676 = stablehlo.convert %arg233 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3677 = stablehlo.dot_general %3675, %3676, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3678 = stablehlo.add %3653, %3677 : tensor<1x8x1024xbf16>
    %3679 = stablehlo.convert %3678 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3680 = stablehlo.multiply %3679, %3679 : tensor<1x8x1024xf32>
    %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3681 = stablehlo.reduce(%3680 init: %cst_164) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3682 = stablehlo.broadcast_in_dim %3681, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3683 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3684 = stablehlo.divide %3682, %3683 : tensor<1x8x1xf32>
    %3685 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3686 = stablehlo.add %3684, %3685 : tensor<1x8x1xf32>
    %3687 = stablehlo.rsqrt %3686 : tensor<1x8x1xf32>
    %3688 = stablehlo.broadcast_in_dim %3687, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3689 = stablehlo.multiply %3679, %3688 : tensor<1x8x1024xf32>
    %3690 = stablehlo.convert %3689 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3691 = stablehlo.convert %arg243 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3692 = stablehlo.broadcast_in_dim %3691, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3693 = stablehlo.broadcast_in_dim %3692, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3694 = stablehlo.multiply %3693, %3690 : tensor<1x8x1024xbf16>
    %3695 = stablehlo.convert %arg252 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3696 = stablehlo.dot_general %3694, %3695, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3697 = stablehlo.convert %arg249 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3698 = stablehlo.dot_general %3694, %3697, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3699 = stablehlo.convert %arg253 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3700 = stablehlo.dot_general %3694, %3699, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3701 = stablehlo.reshape %3696 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3702 = stablehlo.convert %3701 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3703 = stablehlo.multiply %3702, %3702 : tensor<1x8x16x128xf32>
    %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3704 = stablehlo.reduce(%3703 init: %cst_165) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3705 = stablehlo.broadcast_in_dim %3704, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3706 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3707 = stablehlo.divide %3705, %3706 : tensor<1x8x16x1xf32>
    %3708 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3709 = stablehlo.add %3707, %3708 : tensor<1x8x16x1xf32>
    %3710 = stablehlo.rsqrt %3709 : tensor<1x8x16x1xf32>
    %3711 = stablehlo.broadcast_in_dim %3710, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3712 = stablehlo.multiply %3702, %3711 : tensor<1x8x16x128xf32>
    %3713 = stablehlo.convert %3712 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3714 = stablehlo.convert %arg251 : (tensor<128xf32>) -> tensor<128xbf16>
    %3715 = stablehlo.broadcast_in_dim %3714, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3716 = stablehlo.broadcast_in_dim %3715, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3717 = stablehlo.multiply %3716, %3713 : tensor<1x8x16x128xbf16>
    %3718 = stablehlo.reshape %3698 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3719 = stablehlo.convert %3718 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3720 = stablehlo.multiply %3719, %3719 : tensor<1x8x8x128xf32>
    %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3721 = stablehlo.reduce(%3720 init: %cst_166) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3722 = stablehlo.broadcast_in_dim %3721, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3723 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3724 = stablehlo.divide %3722, %3723 : tensor<1x8x8x1xf32>
    %3725 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3726 = stablehlo.add %3724, %3725 : tensor<1x8x8x1xf32>
    %3727 = stablehlo.rsqrt %3726 : tensor<1x8x8x1xf32>
    %3728 = stablehlo.broadcast_in_dim %3727, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3729 = stablehlo.multiply %3719, %3728 : tensor<1x8x8x128xf32>
    %3730 = stablehlo.convert %3729 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3731 = stablehlo.convert %arg248 : (tensor<128xf32>) -> tensor<128xbf16>
    %3732 = stablehlo.broadcast_in_dim %3731, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3733 = stablehlo.broadcast_in_dim %3732, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3734 = stablehlo.multiply %3733, %3730 : tensor<1x8x8x128xbf16>
    %3735 = stablehlo.reshape %3700 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3736 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3737 = stablehlo.compare  LT, %7, %3736,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3738 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3739 = stablehlo.add %7, %3738 : tensor<1x8xi32>
    %3740 = stablehlo.select %3737, %3739, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3741 = stablehlo.broadcast_in_dim %3740, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3742 = "stablehlo.gather"(%26, %3741) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3743 = stablehlo.slice %3742 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3744 = stablehlo.slice %3742 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3745 = stablehlo.broadcast_in_dim %3743, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3746 = stablehlo.convert %3745 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3747 = stablehlo.broadcast_in_dim %3744, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3748 = stablehlo.convert %3747 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3749 = stablehlo.slice %3717 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3750 = stablehlo.slice %3717 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3751 = stablehlo.broadcast_in_dim %3746, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3752 = stablehlo.multiply %3749, %3751 : tensor<1x8x16x64xbf16>
    %3753 = stablehlo.broadcast_in_dim %3748, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3754 = stablehlo.multiply %3750, %3753 : tensor<1x8x16x64xbf16>
    %3755 = stablehlo.subtract %3752, %3754 : tensor<1x8x16x64xbf16>
    %3756 = stablehlo.broadcast_in_dim %3746, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3757 = stablehlo.multiply %3750, %3756 : tensor<1x8x16x64xbf16>
    %3758 = stablehlo.broadcast_in_dim %3748, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3759 = stablehlo.multiply %3749, %3758 : tensor<1x8x16x64xbf16>
    %3760 = stablehlo.add %3757, %3759 : tensor<1x8x16x64xbf16>
    %3761 = stablehlo.concatenate %3755, %3760, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3762 = stablehlo.broadcast_in_dim %3743, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3763 = stablehlo.convert %3762 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3764 = stablehlo.broadcast_in_dim %3744, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3765 = stablehlo.convert %3764 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3766 = stablehlo.slice %3734 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3767 = stablehlo.slice %3734 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3768 = stablehlo.broadcast_in_dim %3763, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3769 = stablehlo.multiply %3766, %3768 : tensor<1x8x8x64xbf16>
    %3770 = stablehlo.broadcast_in_dim %3765, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3771 = stablehlo.multiply %3767, %3770 : tensor<1x8x8x64xbf16>
    %3772 = stablehlo.subtract %3769, %3771 : tensor<1x8x8x64xbf16>
    %3773 = stablehlo.broadcast_in_dim %3763, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3774 = stablehlo.multiply %3767, %3773 : tensor<1x8x8x64xbf16>
    %3775 = stablehlo.broadcast_in_dim %3765, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3776 = stablehlo.multiply %3766, %3775 : tensor<1x8x8x64xbf16>
    %3777 = stablehlo.add %3774, %3776 : tensor<1x8x8x64xbf16>
    %3778 = stablehlo.concatenate %3772, %3777, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3779 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3780 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3781 = stablehlo.broadcast_in_dim %3779, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3782 = stablehlo.and %3781, %3780 : tensor<1x1x8x8xi1>
    %3783 = stablehlo.convert %3782 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3784 = sdy.sharding_constraint %3761 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3785 = sdy.sharding_constraint %3778 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3786 = sdy.sharding_constraint %3735 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3787 = sdy.sharding_constraint %3783 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3788 = stablehlo.reshape %3784 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3789 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3790 = stablehlo.multiply %3788, %3789 : tensor<1x8x8x2x128xbf16>
    %3791 = stablehlo.dot_general %3785, %3790, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3792 = stablehlo.transpose %3791, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_167 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3793 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3794 = stablehlo.compare  NE, %3787, %3793,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3795 = stablehlo.convert %3794 : tensor<1x1x8x8xi1>
    %3796 = stablehlo.broadcast_in_dim %3795, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3797 = stablehlo.reshape %3796 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3798 = call @_where_83(%3797, %3792, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3799 = stablehlo.convert %3798 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_168 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3800 = stablehlo.reduce(%3799 init: %cst_168) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3801 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3802 = stablehlo.maximum %3801, %3800 : tensor<1x8x2x8xf32>
    %3803 = stablehlo.broadcast_in_dim %3802, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3804 = stablehlo.broadcast_in_dim %3803, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3805 = stablehlo.subtract %3799, %3804 : tensor<1x8x2x8x8xf32>
    %3806 = stablehlo.exponential %3805 : tensor<1x8x2x8x8xf32>
    %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3807 = stablehlo.reduce(%3806 init: %cst_169) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3808 = stablehlo.broadcast_in_dim %3807, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3809 = stablehlo.broadcast_in_dim %3808, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3810 = stablehlo.divide %3806, %3809 : tensor<1x8x2x8x8xf32>
    %3811 = stablehlo.convert %3810 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3812 = stablehlo.dot_general %3786, %3811, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3813 = stablehlo.transpose %3812, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3814 = stablehlo.reshape %3813 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3815 = sdy.sharding_constraint %3814 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3816 = stablehlo.reshape %3815 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3817 = stablehlo.convert %arg250 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3818 = stablehlo.dot_general %3816, %3817, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3819 = stablehlo.add %3678, %3818 : tensor<1x8x1024xbf16>
    %3820 = stablehlo.convert %3819 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3821 = stablehlo.multiply %3820, %3820 : tensor<1x8x1024xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3822 = stablehlo.reduce(%3821 init: %cst_170) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3823 = stablehlo.broadcast_in_dim %3822, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3824 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3825 = stablehlo.divide %3823, %3824 : tensor<1x8x1xf32>
    %3826 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3827 = stablehlo.add %3825, %3826 : tensor<1x8x1xf32>
    %3828 = stablehlo.rsqrt %3827 : tensor<1x8x1xf32>
    %3829 = stablehlo.broadcast_in_dim %3828, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3830 = stablehlo.multiply %3820, %3829 : tensor<1x8x1024xf32>
    %3831 = stablehlo.convert %3830 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3832 = stablehlo.convert %arg247 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3833 = stablehlo.broadcast_in_dim %3832, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3834 = stablehlo.broadcast_in_dim %3833, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3835 = stablehlo.multiply %3834, %3831 : tensor<1x8x1024xbf16>
    %3836 = stablehlo.convert %arg245 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3837 = stablehlo.dot_general %3835, %3836, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3838 = call @silu(%3837) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3839 = stablehlo.convert %arg246 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %3840 = stablehlo.dot_general %3835, %3839, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %3841 = stablehlo.multiply %3838, %3840 : tensor<1x8x3072xbf16>
    %3842 = stablehlo.convert %arg244 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %3843 = stablehlo.dot_general %3841, %3842, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3844 = stablehlo.add %3819, %3843 : tensor<1x8x1024xbf16>
    %3845 = stablehlo.convert %3844 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3846 = stablehlo.multiply %3845, %3845 : tensor<1x8x1024xf32>
    %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3847 = stablehlo.reduce(%3846 init: %cst_171) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3848 = stablehlo.broadcast_in_dim %3847, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3849 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3850 = stablehlo.divide %3848, %3849 : tensor<1x8x1xf32>
    %3851 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3852 = stablehlo.add %3850, %3851 : tensor<1x8x1xf32>
    %3853 = stablehlo.rsqrt %3852 : tensor<1x8x1xf32>
    %3854 = stablehlo.broadcast_in_dim %3853, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3855 = stablehlo.multiply %3845, %3854 : tensor<1x8x1024xf32>
    %3856 = stablehlo.convert %3855 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3857 = stablehlo.convert %arg254 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3858 = stablehlo.broadcast_in_dim %3857, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %3859 = stablehlo.broadcast_in_dim %3858, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3860 = stablehlo.multiply %3859, %3856 : tensor<1x8x1024xbf16>
    %3861 = stablehlo.convert %arg263 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %3862 = stablehlo.dot_general %3860, %3861, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %3863 = stablehlo.convert %arg260 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3864 = stablehlo.dot_general %3860, %3863, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3865 = stablehlo.convert %arg264 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %3866 = stablehlo.dot_general %3860, %3865, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3867 = stablehlo.reshape %3862 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %3868 = stablehlo.convert %3867 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %3869 = stablehlo.multiply %3868, %3868 : tensor<1x8x16x128xf32>
    %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3870 = stablehlo.reduce(%3869 init: %cst_172) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %3871 = stablehlo.broadcast_in_dim %3870, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %3872 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3873 = stablehlo.divide %3871, %3872 : tensor<1x8x16x1xf32>
    %3874 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %3875 = stablehlo.add %3873, %3874 : tensor<1x8x16x1xf32>
    %3876 = stablehlo.rsqrt %3875 : tensor<1x8x16x1xf32>
    %3877 = stablehlo.broadcast_in_dim %3876, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %3878 = stablehlo.multiply %3868, %3877 : tensor<1x8x16x128xf32>
    %3879 = stablehlo.convert %3878 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %3880 = stablehlo.convert %arg262 : (tensor<128xf32>) -> tensor<128xbf16>
    %3881 = stablehlo.broadcast_in_dim %3880, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3882 = stablehlo.broadcast_in_dim %3881, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3883 = stablehlo.multiply %3882, %3879 : tensor<1x8x16x128xbf16>
    %3884 = stablehlo.reshape %3864 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3885 = stablehlo.convert %3884 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %3886 = stablehlo.multiply %3885, %3885 : tensor<1x8x8x128xf32>
    %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3887 = stablehlo.reduce(%3886 init: %cst_173) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %3888 = stablehlo.broadcast_in_dim %3887, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %3889 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3890 = stablehlo.divide %3888, %3889 : tensor<1x8x8x1xf32>
    %3891 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %3892 = stablehlo.add %3890, %3891 : tensor<1x8x8x1xf32>
    %3893 = stablehlo.rsqrt %3892 : tensor<1x8x8x1xf32>
    %3894 = stablehlo.broadcast_in_dim %3893, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %3895 = stablehlo.multiply %3885, %3894 : tensor<1x8x8x128xf32>
    %3896 = stablehlo.convert %3895 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %3897 = stablehlo.convert %arg259 : (tensor<128xf32>) -> tensor<128xbf16>
    %3898 = stablehlo.broadcast_in_dim %3897, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %3899 = stablehlo.broadcast_in_dim %3898, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %3900 = stablehlo.multiply %3899, %3896 : tensor<1x8x8x128xbf16>
    %3901 = stablehlo.reshape %3866 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %3902 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3903 = stablehlo.compare  LT, %7, %3902,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %3904 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %3905 = stablehlo.add %7, %3904 : tensor<1x8xi32>
    %3906 = stablehlo.select %3903, %3905, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %3907 = stablehlo.broadcast_in_dim %3906, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %3908 = "stablehlo.gather"(%26, %3907) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %3909 = stablehlo.slice %3908 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3910 = stablehlo.slice %3908 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %3911 = stablehlo.broadcast_in_dim %3909, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3912 = stablehlo.convert %3911 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3913 = stablehlo.broadcast_in_dim %3910, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3914 = stablehlo.convert %3913 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3915 = stablehlo.slice %3883 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3916 = stablehlo.slice %3883 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %3917 = stablehlo.broadcast_in_dim %3912, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3918 = stablehlo.multiply %3915, %3917 : tensor<1x8x16x64xbf16>
    %3919 = stablehlo.broadcast_in_dim %3914, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3920 = stablehlo.multiply %3916, %3919 : tensor<1x8x16x64xbf16>
    %3921 = stablehlo.subtract %3918, %3920 : tensor<1x8x16x64xbf16>
    %3922 = stablehlo.broadcast_in_dim %3912, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3923 = stablehlo.multiply %3916, %3922 : tensor<1x8x16x64xbf16>
    %3924 = stablehlo.broadcast_in_dim %3914, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %3925 = stablehlo.multiply %3915, %3924 : tensor<1x8x16x64xbf16>
    %3926 = stablehlo.add %3923, %3925 : tensor<1x8x16x64xbf16>
    %3927 = stablehlo.concatenate %3921, %3926, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %3928 = stablehlo.broadcast_in_dim %3909, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3929 = stablehlo.convert %3928 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3930 = stablehlo.broadcast_in_dim %3910, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %3931 = stablehlo.convert %3930 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %3932 = stablehlo.slice %3900 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3933 = stablehlo.slice %3900 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %3934 = stablehlo.broadcast_in_dim %3929, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3935 = stablehlo.multiply %3932, %3934 : tensor<1x8x8x64xbf16>
    %3936 = stablehlo.broadcast_in_dim %3931, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3937 = stablehlo.multiply %3933, %3936 : tensor<1x8x8x64xbf16>
    %3938 = stablehlo.subtract %3935, %3937 : tensor<1x8x8x64xbf16>
    %3939 = stablehlo.broadcast_in_dim %3929, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3940 = stablehlo.multiply %3933, %3939 : tensor<1x8x8x64xbf16>
    %3941 = stablehlo.broadcast_in_dim %3931, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %3942 = stablehlo.multiply %3932, %3941 : tensor<1x8x8x64xbf16>
    %3943 = stablehlo.add %3940, %3942 : tensor<1x8x8x64xbf16>
    %3944 = stablehlo.concatenate %3938, %3943, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %3945 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %3946 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %3947 = stablehlo.broadcast_in_dim %3945, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %3948 = stablehlo.and %3947, %3946 : tensor<1x1x8x8xi1>
    %3949 = stablehlo.convert %3948 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %3950 = sdy.sharding_constraint %3927 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3951 = sdy.sharding_constraint %3944 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3952 = sdy.sharding_constraint %3901 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %3953 = sdy.sharding_constraint %3949 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %3954 = stablehlo.reshape %3950 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3955 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %3956 = stablehlo.multiply %3954, %3955 : tensor<1x8x8x2x128xbf16>
    %3957 = stablehlo.dot_general %3951, %3956, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %3958 = stablehlo.transpose %3957, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3959 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %3960 = stablehlo.compare  NE, %3953, %3959,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %3961 = stablehlo.convert %3960 : tensor<1x1x8x8xi1>
    %3962 = stablehlo.broadcast_in_dim %3961, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %3963 = stablehlo.reshape %3962 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %3964 = call @_where_83(%3963, %3958, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %3965 = stablehlo.convert %3964 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_175 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3966 = stablehlo.reduce(%3965 init: %cst_175) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3967 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %3968 = stablehlo.maximum %3967, %3966 : tensor<1x8x2x8xf32>
    %3969 = stablehlo.broadcast_in_dim %3968, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3970 = stablehlo.broadcast_in_dim %3969, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3971 = stablehlo.subtract %3965, %3970 : tensor<1x8x2x8x8xf32>
    %3972 = stablehlo.exponential %3971 : tensor<1x8x2x8x8xf32>
    %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3973 = stablehlo.reduce(%3972 init: %cst_176) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %3974 = stablehlo.broadcast_in_dim %3973, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %3975 = stablehlo.broadcast_in_dim %3974, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %3976 = stablehlo.divide %3972, %3975 : tensor<1x8x2x8x8xf32>
    %3977 = stablehlo.convert %3976 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %3978 = stablehlo.dot_general %3952, %3977, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %3979 = stablehlo.transpose %3978, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %3980 = stablehlo.reshape %3979 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %3981 = sdy.sharding_constraint %3980 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %3982 = stablehlo.reshape %3981 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %3983 = stablehlo.convert %arg261 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %3984 = stablehlo.dot_general %3982, %3983, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %3985 = stablehlo.add %3844, %3984 : tensor<1x8x1024xbf16>
    %3986 = stablehlo.convert %3985 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %3987 = stablehlo.multiply %3986, %3986 : tensor<1x8x1024xf32>
    %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3988 = stablehlo.reduce(%3987 init: %cst_177) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3989 = stablehlo.broadcast_in_dim %3988, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %3990 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3991 = stablehlo.divide %3989, %3990 : tensor<1x8x1xf32>
    %3992 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %3993 = stablehlo.add %3991, %3992 : tensor<1x8x1xf32>
    %3994 = stablehlo.rsqrt %3993 : tensor<1x8x1xf32>
    %3995 = stablehlo.broadcast_in_dim %3994, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %3996 = stablehlo.multiply %3986, %3995 : tensor<1x8x1024xf32>
    %3997 = stablehlo.convert %3996 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %3998 = stablehlo.convert %arg258 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %3999 = stablehlo.broadcast_in_dim %3998, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4000 = stablehlo.broadcast_in_dim %3999, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4001 = stablehlo.multiply %4000, %3997 : tensor<1x8x1024xbf16>
    %4002 = stablehlo.convert %arg256 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4003 = stablehlo.dot_general %4001, %4002, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4004 = call @silu(%4003) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4005 = stablehlo.convert %arg257 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4006 = stablehlo.dot_general %4001, %4005, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4007 = stablehlo.multiply %4004, %4006 : tensor<1x8x3072xbf16>
    %4008 = stablehlo.convert %arg255 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4009 = stablehlo.dot_general %4007, %4008, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4010 = stablehlo.add %3985, %4009 : tensor<1x8x1024xbf16>
    %4011 = stablehlo.convert %4010 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4012 = stablehlo.multiply %4011, %4011 : tensor<1x8x1024xf32>
    %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4013 = stablehlo.reduce(%4012 init: %cst_178) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4014 = stablehlo.broadcast_in_dim %4013, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4015 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4016 = stablehlo.divide %4014, %4015 : tensor<1x8x1xf32>
    %4017 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4018 = stablehlo.add %4016, %4017 : tensor<1x8x1xf32>
    %4019 = stablehlo.rsqrt %4018 : tensor<1x8x1xf32>
    %4020 = stablehlo.broadcast_in_dim %4019, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4021 = stablehlo.multiply %4011, %4020 : tensor<1x8x1024xf32>
    %4022 = stablehlo.convert %4021 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4023 = stablehlo.convert %arg265 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4024 = stablehlo.broadcast_in_dim %4023, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4025 = stablehlo.broadcast_in_dim %4024, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4026 = stablehlo.multiply %4025, %4022 : tensor<1x8x1024xbf16>
    %4027 = stablehlo.convert %arg274 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4028 = stablehlo.dot_general %4026, %4027, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4029 = stablehlo.convert %arg271 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4030 = stablehlo.dot_general %4026, %4029, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4031 = stablehlo.convert %arg275 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4032 = stablehlo.dot_general %4026, %4031, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4033 = stablehlo.reshape %4028 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4034 = stablehlo.convert %4033 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4035 = stablehlo.multiply %4034, %4034 : tensor<1x8x16x128xf32>
    %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4036 = stablehlo.reduce(%4035 init: %cst_179) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4037 = stablehlo.broadcast_in_dim %4036, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4038 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4039 = stablehlo.divide %4037, %4038 : tensor<1x8x16x1xf32>
    %4040 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4041 = stablehlo.add %4039, %4040 : tensor<1x8x16x1xf32>
    %4042 = stablehlo.rsqrt %4041 : tensor<1x8x16x1xf32>
    %4043 = stablehlo.broadcast_in_dim %4042, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4044 = stablehlo.multiply %4034, %4043 : tensor<1x8x16x128xf32>
    %4045 = stablehlo.convert %4044 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4046 = stablehlo.convert %arg273 : (tensor<128xf32>) -> tensor<128xbf16>
    %4047 = stablehlo.broadcast_in_dim %4046, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4048 = stablehlo.broadcast_in_dim %4047, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4049 = stablehlo.multiply %4048, %4045 : tensor<1x8x16x128xbf16>
    %4050 = stablehlo.reshape %4030 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4051 = stablehlo.convert %4050 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4052 = stablehlo.multiply %4051, %4051 : tensor<1x8x8x128xf32>
    %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4053 = stablehlo.reduce(%4052 init: %cst_180) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4054 = stablehlo.broadcast_in_dim %4053, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4055 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4056 = stablehlo.divide %4054, %4055 : tensor<1x8x8x1xf32>
    %4057 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4058 = stablehlo.add %4056, %4057 : tensor<1x8x8x1xf32>
    %4059 = stablehlo.rsqrt %4058 : tensor<1x8x8x1xf32>
    %4060 = stablehlo.broadcast_in_dim %4059, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4061 = stablehlo.multiply %4051, %4060 : tensor<1x8x8x128xf32>
    %4062 = stablehlo.convert %4061 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4063 = stablehlo.convert %arg270 : (tensor<128xf32>) -> tensor<128xbf16>
    %4064 = stablehlo.broadcast_in_dim %4063, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4065 = stablehlo.broadcast_in_dim %4064, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4066 = stablehlo.multiply %4065, %4062 : tensor<1x8x8x128xbf16>
    %4067 = stablehlo.reshape %4032 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4068 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4069 = stablehlo.compare  LT, %7, %4068,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4070 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4071 = stablehlo.add %7, %4070 : tensor<1x8xi32>
    %4072 = stablehlo.select %4069, %4071, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4073 = stablehlo.broadcast_in_dim %4072, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4074 = "stablehlo.gather"(%26, %4073) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4075 = stablehlo.slice %4074 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4076 = stablehlo.slice %4074 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4077 = stablehlo.broadcast_in_dim %4075, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4078 = stablehlo.convert %4077 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4079 = stablehlo.broadcast_in_dim %4076, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4080 = stablehlo.convert %4079 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4081 = stablehlo.slice %4049 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4082 = stablehlo.slice %4049 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4083 = stablehlo.broadcast_in_dim %4078, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4084 = stablehlo.multiply %4081, %4083 : tensor<1x8x16x64xbf16>
    %4085 = stablehlo.broadcast_in_dim %4080, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4086 = stablehlo.multiply %4082, %4085 : tensor<1x8x16x64xbf16>
    %4087 = stablehlo.subtract %4084, %4086 : tensor<1x8x16x64xbf16>
    %4088 = stablehlo.broadcast_in_dim %4078, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4089 = stablehlo.multiply %4082, %4088 : tensor<1x8x16x64xbf16>
    %4090 = stablehlo.broadcast_in_dim %4080, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4091 = stablehlo.multiply %4081, %4090 : tensor<1x8x16x64xbf16>
    %4092 = stablehlo.add %4089, %4091 : tensor<1x8x16x64xbf16>
    %4093 = stablehlo.concatenate %4087, %4092, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4094 = stablehlo.broadcast_in_dim %4075, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4095 = stablehlo.convert %4094 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4096 = stablehlo.broadcast_in_dim %4076, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4097 = stablehlo.convert %4096 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4098 = stablehlo.slice %4066 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4099 = stablehlo.slice %4066 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4100 = stablehlo.broadcast_in_dim %4095, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4101 = stablehlo.multiply %4098, %4100 : tensor<1x8x8x64xbf16>
    %4102 = stablehlo.broadcast_in_dim %4097, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4103 = stablehlo.multiply %4099, %4102 : tensor<1x8x8x64xbf16>
    %4104 = stablehlo.subtract %4101, %4103 : tensor<1x8x8x64xbf16>
    %4105 = stablehlo.broadcast_in_dim %4095, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4106 = stablehlo.multiply %4099, %4105 : tensor<1x8x8x64xbf16>
    %4107 = stablehlo.broadcast_in_dim %4097, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4108 = stablehlo.multiply %4098, %4107 : tensor<1x8x8x64xbf16>
    %4109 = stablehlo.add %4106, %4108 : tensor<1x8x8x64xbf16>
    %4110 = stablehlo.concatenate %4104, %4109, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4111 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4112 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4113 = stablehlo.broadcast_in_dim %4111, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4114 = stablehlo.and %4113, %4112 : tensor<1x1x8x8xi1>
    %4115 = stablehlo.convert %4114 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4116 = sdy.sharding_constraint %4093 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4117 = sdy.sharding_constraint %4110 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4118 = sdy.sharding_constraint %4067 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4119 = sdy.sharding_constraint %4115 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4120 = stablehlo.reshape %4116 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4121 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4122 = stablehlo.multiply %4120, %4121 : tensor<1x8x8x2x128xbf16>
    %4123 = stablehlo.dot_general %4117, %4122, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4124 = stablehlo.transpose %4123, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_181 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4125 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4126 = stablehlo.compare  NE, %4119, %4125,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4127 = stablehlo.convert %4126 : tensor<1x1x8x8xi1>
    %4128 = stablehlo.broadcast_in_dim %4127, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4129 = stablehlo.reshape %4128 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4130 = call @_where_83(%4129, %4124, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4131 = stablehlo.convert %4130 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_182 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4132 = stablehlo.reduce(%4131 init: %cst_182) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4133 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4134 = stablehlo.maximum %4133, %4132 : tensor<1x8x2x8xf32>
    %4135 = stablehlo.broadcast_in_dim %4134, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4136 = stablehlo.broadcast_in_dim %4135, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4137 = stablehlo.subtract %4131, %4136 : tensor<1x8x2x8x8xf32>
    %4138 = stablehlo.exponential %4137 : tensor<1x8x2x8x8xf32>
    %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4139 = stablehlo.reduce(%4138 init: %cst_183) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4140 = stablehlo.broadcast_in_dim %4139, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4141 = stablehlo.broadcast_in_dim %4140, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4142 = stablehlo.divide %4138, %4141 : tensor<1x8x2x8x8xf32>
    %4143 = stablehlo.convert %4142 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4144 = stablehlo.dot_general %4118, %4143, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4145 = stablehlo.transpose %4144, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4146 = stablehlo.reshape %4145 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4147 = sdy.sharding_constraint %4146 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4148 = stablehlo.reshape %4147 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4149 = stablehlo.convert %arg272 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4150 = stablehlo.dot_general %4148, %4149, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4151 = stablehlo.add %4010, %4150 : tensor<1x8x1024xbf16>
    %4152 = stablehlo.convert %4151 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4153 = stablehlo.multiply %4152, %4152 : tensor<1x8x1024xf32>
    %cst_184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4154 = stablehlo.reduce(%4153 init: %cst_184) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4155 = stablehlo.broadcast_in_dim %4154, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4156 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4157 = stablehlo.divide %4155, %4156 : tensor<1x8x1xf32>
    %4158 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4159 = stablehlo.add %4157, %4158 : tensor<1x8x1xf32>
    %4160 = stablehlo.rsqrt %4159 : tensor<1x8x1xf32>
    %4161 = stablehlo.broadcast_in_dim %4160, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4162 = stablehlo.multiply %4152, %4161 : tensor<1x8x1024xf32>
    %4163 = stablehlo.convert %4162 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4164 = stablehlo.convert %arg269 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4165 = stablehlo.broadcast_in_dim %4164, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4166 = stablehlo.broadcast_in_dim %4165, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4167 = stablehlo.multiply %4166, %4163 : tensor<1x8x1024xbf16>
    %4168 = stablehlo.convert %arg267 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4169 = stablehlo.dot_general %4167, %4168, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4170 = call @silu(%4169) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4171 = stablehlo.convert %arg268 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4172 = stablehlo.dot_general %4167, %4171, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4173 = stablehlo.multiply %4170, %4172 : tensor<1x8x3072xbf16>
    %4174 = stablehlo.convert %arg266 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4175 = stablehlo.dot_general %4173, %4174, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4176 = stablehlo.add %4151, %4175 : tensor<1x8x1024xbf16>
    %4177 = stablehlo.convert %4176 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4178 = stablehlo.multiply %4177, %4177 : tensor<1x8x1024xf32>
    %cst_185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4179 = stablehlo.reduce(%4178 init: %cst_185) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4180 = stablehlo.broadcast_in_dim %4179, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4181 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4182 = stablehlo.divide %4180, %4181 : tensor<1x8x1xf32>
    %4183 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4184 = stablehlo.add %4182, %4183 : tensor<1x8x1xf32>
    %4185 = stablehlo.rsqrt %4184 : tensor<1x8x1xf32>
    %4186 = stablehlo.broadcast_in_dim %4185, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4187 = stablehlo.multiply %4177, %4186 : tensor<1x8x1024xf32>
    %4188 = stablehlo.convert %4187 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4189 = stablehlo.convert %arg276 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4190 = stablehlo.broadcast_in_dim %4189, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4191 = stablehlo.broadcast_in_dim %4190, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4192 = stablehlo.multiply %4191, %4188 : tensor<1x8x1024xbf16>
    %4193 = stablehlo.convert %arg285 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4194 = stablehlo.dot_general %4192, %4193, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4195 = stablehlo.convert %arg282 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4196 = stablehlo.dot_general %4192, %4195, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4197 = stablehlo.convert %arg286 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4198 = stablehlo.dot_general %4192, %4197, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4199 = stablehlo.reshape %4194 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4200 = stablehlo.convert %4199 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4201 = stablehlo.multiply %4200, %4200 : tensor<1x8x16x128xf32>
    %cst_186 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4202 = stablehlo.reduce(%4201 init: %cst_186) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4203 = stablehlo.broadcast_in_dim %4202, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4204 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4205 = stablehlo.divide %4203, %4204 : tensor<1x8x16x1xf32>
    %4206 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4207 = stablehlo.add %4205, %4206 : tensor<1x8x16x1xf32>
    %4208 = stablehlo.rsqrt %4207 : tensor<1x8x16x1xf32>
    %4209 = stablehlo.broadcast_in_dim %4208, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4210 = stablehlo.multiply %4200, %4209 : tensor<1x8x16x128xf32>
    %4211 = stablehlo.convert %4210 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4212 = stablehlo.convert %arg284 : (tensor<128xf32>) -> tensor<128xbf16>
    %4213 = stablehlo.broadcast_in_dim %4212, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4214 = stablehlo.broadcast_in_dim %4213, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4215 = stablehlo.multiply %4214, %4211 : tensor<1x8x16x128xbf16>
    %4216 = stablehlo.reshape %4196 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4217 = stablehlo.convert %4216 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4218 = stablehlo.multiply %4217, %4217 : tensor<1x8x8x128xf32>
    %cst_187 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4219 = stablehlo.reduce(%4218 init: %cst_187) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4220 = stablehlo.broadcast_in_dim %4219, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4221 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4222 = stablehlo.divide %4220, %4221 : tensor<1x8x8x1xf32>
    %4223 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4224 = stablehlo.add %4222, %4223 : tensor<1x8x8x1xf32>
    %4225 = stablehlo.rsqrt %4224 : tensor<1x8x8x1xf32>
    %4226 = stablehlo.broadcast_in_dim %4225, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4227 = stablehlo.multiply %4217, %4226 : tensor<1x8x8x128xf32>
    %4228 = stablehlo.convert %4227 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4229 = stablehlo.convert %arg281 : (tensor<128xf32>) -> tensor<128xbf16>
    %4230 = stablehlo.broadcast_in_dim %4229, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4231 = stablehlo.broadcast_in_dim %4230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4232 = stablehlo.multiply %4231, %4228 : tensor<1x8x8x128xbf16>
    %4233 = stablehlo.reshape %4198 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4234 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4235 = stablehlo.compare  LT, %7, %4234,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4236 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4237 = stablehlo.add %7, %4236 : tensor<1x8xi32>
    %4238 = stablehlo.select %4235, %4237, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4239 = stablehlo.broadcast_in_dim %4238, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4240 = "stablehlo.gather"(%26, %4239) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4241 = stablehlo.slice %4240 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4242 = stablehlo.slice %4240 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4243 = stablehlo.broadcast_in_dim %4241, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4244 = stablehlo.convert %4243 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4245 = stablehlo.broadcast_in_dim %4242, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4246 = stablehlo.convert %4245 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4247 = stablehlo.slice %4215 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4248 = stablehlo.slice %4215 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4249 = stablehlo.broadcast_in_dim %4244, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4250 = stablehlo.multiply %4247, %4249 : tensor<1x8x16x64xbf16>
    %4251 = stablehlo.broadcast_in_dim %4246, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4252 = stablehlo.multiply %4248, %4251 : tensor<1x8x16x64xbf16>
    %4253 = stablehlo.subtract %4250, %4252 : tensor<1x8x16x64xbf16>
    %4254 = stablehlo.broadcast_in_dim %4244, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4255 = stablehlo.multiply %4248, %4254 : tensor<1x8x16x64xbf16>
    %4256 = stablehlo.broadcast_in_dim %4246, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4257 = stablehlo.multiply %4247, %4256 : tensor<1x8x16x64xbf16>
    %4258 = stablehlo.add %4255, %4257 : tensor<1x8x16x64xbf16>
    %4259 = stablehlo.concatenate %4253, %4258, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4260 = stablehlo.broadcast_in_dim %4241, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4261 = stablehlo.convert %4260 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4262 = stablehlo.broadcast_in_dim %4242, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4263 = stablehlo.convert %4262 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4264 = stablehlo.slice %4232 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4265 = stablehlo.slice %4232 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4266 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4267 = stablehlo.multiply %4264, %4266 : tensor<1x8x8x64xbf16>
    %4268 = stablehlo.broadcast_in_dim %4263, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4269 = stablehlo.multiply %4265, %4268 : tensor<1x8x8x64xbf16>
    %4270 = stablehlo.subtract %4267, %4269 : tensor<1x8x8x64xbf16>
    %4271 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4272 = stablehlo.multiply %4265, %4271 : tensor<1x8x8x64xbf16>
    %4273 = stablehlo.broadcast_in_dim %4263, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4274 = stablehlo.multiply %4264, %4273 : tensor<1x8x8x64xbf16>
    %4275 = stablehlo.add %4272, %4274 : tensor<1x8x8x64xbf16>
    %4276 = stablehlo.concatenate %4270, %4275, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4277 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4278 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4279 = stablehlo.broadcast_in_dim %4277, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4280 = stablehlo.and %4279, %4278 : tensor<1x1x8x8xi1>
    %4281 = stablehlo.convert %4280 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4282 = sdy.sharding_constraint %4259 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4283 = sdy.sharding_constraint %4276 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4284 = sdy.sharding_constraint %4233 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4285 = sdy.sharding_constraint %4281 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4286 = stablehlo.reshape %4282 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4287 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4288 = stablehlo.multiply %4286, %4287 : tensor<1x8x8x2x128xbf16>
    %4289 = stablehlo.dot_general %4283, %4288, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4290 = stablehlo.transpose %4289, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_188 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4291 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4292 = stablehlo.compare  NE, %4285, %4291,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4293 = stablehlo.convert %4292 : tensor<1x1x8x8xi1>
    %4294 = stablehlo.broadcast_in_dim %4293, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4295 = stablehlo.reshape %4294 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4296 = call @_where_83(%4295, %4290, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4297 = stablehlo.convert %4296 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_189 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4298 = stablehlo.reduce(%4297 init: %cst_189) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4299 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4300 = stablehlo.maximum %4299, %4298 : tensor<1x8x2x8xf32>
    %4301 = stablehlo.broadcast_in_dim %4300, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4302 = stablehlo.broadcast_in_dim %4301, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4303 = stablehlo.subtract %4297, %4302 : tensor<1x8x2x8x8xf32>
    %4304 = stablehlo.exponential %4303 : tensor<1x8x2x8x8xf32>
    %cst_190 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4305 = stablehlo.reduce(%4304 init: %cst_190) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4306 = stablehlo.broadcast_in_dim %4305, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4307 = stablehlo.broadcast_in_dim %4306, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4308 = stablehlo.divide %4304, %4307 : tensor<1x8x2x8x8xf32>
    %4309 = stablehlo.convert %4308 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4310 = stablehlo.dot_general %4284, %4309, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4311 = stablehlo.transpose %4310, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4312 = stablehlo.reshape %4311 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4313 = sdy.sharding_constraint %4312 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4314 = stablehlo.reshape %4313 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4315 = stablehlo.convert %arg283 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4316 = stablehlo.dot_general %4314, %4315, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4317 = stablehlo.add %4176, %4316 : tensor<1x8x1024xbf16>
    %4318 = stablehlo.convert %4317 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4319 = stablehlo.multiply %4318, %4318 : tensor<1x8x1024xf32>
    %cst_191 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4320 = stablehlo.reduce(%4319 init: %cst_191) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4321 = stablehlo.broadcast_in_dim %4320, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4322 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4323 = stablehlo.divide %4321, %4322 : tensor<1x8x1xf32>
    %4324 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4325 = stablehlo.add %4323, %4324 : tensor<1x8x1xf32>
    %4326 = stablehlo.rsqrt %4325 : tensor<1x8x1xf32>
    %4327 = stablehlo.broadcast_in_dim %4326, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4328 = stablehlo.multiply %4318, %4327 : tensor<1x8x1024xf32>
    %4329 = stablehlo.convert %4328 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4330 = stablehlo.convert %arg280 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4331 = stablehlo.broadcast_in_dim %4330, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4332 = stablehlo.broadcast_in_dim %4331, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4333 = stablehlo.multiply %4332, %4329 : tensor<1x8x1024xbf16>
    %4334 = stablehlo.convert %arg278 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4335 = stablehlo.dot_general %4333, %4334, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4336 = call @silu(%4335) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4337 = stablehlo.convert %arg279 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4338 = stablehlo.dot_general %4333, %4337, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4339 = stablehlo.multiply %4336, %4338 : tensor<1x8x3072xbf16>
    %4340 = stablehlo.convert %arg277 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4341 = stablehlo.dot_general %4339, %4340, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4342 = stablehlo.add %4317, %4341 : tensor<1x8x1024xbf16>
    %4343 = stablehlo.convert %4342 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4344 = stablehlo.multiply %4343, %4343 : tensor<1x8x1024xf32>
    %cst_192 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4345 = stablehlo.reduce(%4344 init: %cst_192) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4346 = stablehlo.broadcast_in_dim %4345, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4347 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4348 = stablehlo.divide %4346, %4347 : tensor<1x8x1xf32>
    %4349 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4350 = stablehlo.add %4348, %4349 : tensor<1x8x1xf32>
    %4351 = stablehlo.rsqrt %4350 : tensor<1x8x1xf32>
    %4352 = stablehlo.broadcast_in_dim %4351, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4353 = stablehlo.multiply %4343, %4352 : tensor<1x8x1024xf32>
    %4354 = stablehlo.convert %4353 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4355 = stablehlo.convert %arg287 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4356 = stablehlo.broadcast_in_dim %4355, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4357 = stablehlo.broadcast_in_dim %4356, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4358 = stablehlo.multiply %4357, %4354 : tensor<1x8x1024xbf16>
    %4359 = stablehlo.convert %arg296 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4360 = stablehlo.dot_general %4358, %4359, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4361 = stablehlo.convert %arg293 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4362 = stablehlo.dot_general %4358, %4361, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4363 = stablehlo.convert %arg297 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4364 = stablehlo.dot_general %4358, %4363, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4365 = stablehlo.reshape %4360 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4366 = stablehlo.convert %4365 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4367 = stablehlo.multiply %4366, %4366 : tensor<1x8x16x128xf32>
    %cst_193 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4368 = stablehlo.reduce(%4367 init: %cst_193) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4369 = stablehlo.broadcast_in_dim %4368, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4370 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4371 = stablehlo.divide %4369, %4370 : tensor<1x8x16x1xf32>
    %4372 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4373 = stablehlo.add %4371, %4372 : tensor<1x8x16x1xf32>
    %4374 = stablehlo.rsqrt %4373 : tensor<1x8x16x1xf32>
    %4375 = stablehlo.broadcast_in_dim %4374, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4376 = stablehlo.multiply %4366, %4375 : tensor<1x8x16x128xf32>
    %4377 = stablehlo.convert %4376 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4378 = stablehlo.convert %arg295 : (tensor<128xf32>) -> tensor<128xbf16>
    %4379 = stablehlo.broadcast_in_dim %4378, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4380 = stablehlo.broadcast_in_dim %4379, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4381 = stablehlo.multiply %4380, %4377 : tensor<1x8x16x128xbf16>
    %4382 = stablehlo.reshape %4362 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4383 = stablehlo.convert %4382 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4384 = stablehlo.multiply %4383, %4383 : tensor<1x8x8x128xf32>
    %cst_194 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4385 = stablehlo.reduce(%4384 init: %cst_194) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4386 = stablehlo.broadcast_in_dim %4385, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4387 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4388 = stablehlo.divide %4386, %4387 : tensor<1x8x8x1xf32>
    %4389 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4390 = stablehlo.add %4388, %4389 : tensor<1x8x8x1xf32>
    %4391 = stablehlo.rsqrt %4390 : tensor<1x8x8x1xf32>
    %4392 = stablehlo.broadcast_in_dim %4391, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4393 = stablehlo.multiply %4383, %4392 : tensor<1x8x8x128xf32>
    %4394 = stablehlo.convert %4393 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4395 = stablehlo.convert %arg292 : (tensor<128xf32>) -> tensor<128xbf16>
    %4396 = stablehlo.broadcast_in_dim %4395, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4397 = stablehlo.broadcast_in_dim %4396, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4398 = stablehlo.multiply %4397, %4394 : tensor<1x8x8x128xbf16>
    %4399 = stablehlo.reshape %4364 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4400 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4401 = stablehlo.compare  LT, %7, %4400,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4402 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4403 = stablehlo.add %7, %4402 : tensor<1x8xi32>
    %4404 = stablehlo.select %4401, %4403, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4405 = stablehlo.broadcast_in_dim %4404, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4406 = "stablehlo.gather"(%26, %4405) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4407 = stablehlo.slice %4406 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4408 = stablehlo.slice %4406 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4409 = stablehlo.broadcast_in_dim %4407, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4410 = stablehlo.convert %4409 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4411 = stablehlo.broadcast_in_dim %4408, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4412 = stablehlo.convert %4411 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4413 = stablehlo.slice %4381 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4414 = stablehlo.slice %4381 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4415 = stablehlo.broadcast_in_dim %4410, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4416 = stablehlo.multiply %4413, %4415 : tensor<1x8x16x64xbf16>
    %4417 = stablehlo.broadcast_in_dim %4412, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4418 = stablehlo.multiply %4414, %4417 : tensor<1x8x16x64xbf16>
    %4419 = stablehlo.subtract %4416, %4418 : tensor<1x8x16x64xbf16>
    %4420 = stablehlo.broadcast_in_dim %4410, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4421 = stablehlo.multiply %4414, %4420 : tensor<1x8x16x64xbf16>
    %4422 = stablehlo.broadcast_in_dim %4412, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4423 = stablehlo.multiply %4413, %4422 : tensor<1x8x16x64xbf16>
    %4424 = stablehlo.add %4421, %4423 : tensor<1x8x16x64xbf16>
    %4425 = stablehlo.concatenate %4419, %4424, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4426 = stablehlo.broadcast_in_dim %4407, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4427 = stablehlo.convert %4426 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4428 = stablehlo.broadcast_in_dim %4408, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4429 = stablehlo.convert %4428 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4430 = stablehlo.slice %4398 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4431 = stablehlo.slice %4398 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4432 = stablehlo.broadcast_in_dim %4427, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4433 = stablehlo.multiply %4430, %4432 : tensor<1x8x8x64xbf16>
    %4434 = stablehlo.broadcast_in_dim %4429, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4435 = stablehlo.multiply %4431, %4434 : tensor<1x8x8x64xbf16>
    %4436 = stablehlo.subtract %4433, %4435 : tensor<1x8x8x64xbf16>
    %4437 = stablehlo.broadcast_in_dim %4427, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4438 = stablehlo.multiply %4431, %4437 : tensor<1x8x8x64xbf16>
    %4439 = stablehlo.broadcast_in_dim %4429, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4440 = stablehlo.multiply %4430, %4439 : tensor<1x8x8x64xbf16>
    %4441 = stablehlo.add %4438, %4440 : tensor<1x8x8x64xbf16>
    %4442 = stablehlo.concatenate %4436, %4441, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4443 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4444 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4445 = stablehlo.broadcast_in_dim %4443, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4446 = stablehlo.and %4445, %4444 : tensor<1x1x8x8xi1>
    %4447 = stablehlo.convert %4446 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4448 = sdy.sharding_constraint %4425 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4449 = sdy.sharding_constraint %4442 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4450 = sdy.sharding_constraint %4399 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4451 = sdy.sharding_constraint %4447 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4452 = stablehlo.reshape %4448 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4453 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4454 = stablehlo.multiply %4452, %4453 : tensor<1x8x8x2x128xbf16>
    %4455 = stablehlo.dot_general %4449, %4454, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4456 = stablehlo.transpose %4455, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_195 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4457 = stablehlo.broadcast_in_dim %cst_195, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4458 = stablehlo.compare  NE, %4451, %4457,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4459 = stablehlo.convert %4458 : tensor<1x1x8x8xi1>
    %4460 = stablehlo.broadcast_in_dim %4459, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4461 = stablehlo.reshape %4460 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4462 = call @_where_83(%4461, %4456, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4463 = stablehlo.convert %4462 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_196 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4464 = stablehlo.reduce(%4463 init: %cst_196) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4465 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4466 = stablehlo.maximum %4465, %4464 : tensor<1x8x2x8xf32>
    %4467 = stablehlo.broadcast_in_dim %4466, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4468 = stablehlo.broadcast_in_dim %4467, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4469 = stablehlo.subtract %4463, %4468 : tensor<1x8x2x8x8xf32>
    %4470 = stablehlo.exponential %4469 : tensor<1x8x2x8x8xf32>
    %cst_197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4471 = stablehlo.reduce(%4470 init: %cst_197) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4472 = stablehlo.broadcast_in_dim %4471, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4473 = stablehlo.broadcast_in_dim %4472, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4474 = stablehlo.divide %4470, %4473 : tensor<1x8x2x8x8xf32>
    %4475 = stablehlo.convert %4474 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4476 = stablehlo.dot_general %4450, %4475, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4477 = stablehlo.transpose %4476, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4478 = stablehlo.reshape %4477 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4479 = sdy.sharding_constraint %4478 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4480 = stablehlo.reshape %4479 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4481 = stablehlo.convert %arg294 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4482 = stablehlo.dot_general %4480, %4481, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4483 = stablehlo.add %4342, %4482 : tensor<1x8x1024xbf16>
    %4484 = stablehlo.convert %4483 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4485 = stablehlo.multiply %4484, %4484 : tensor<1x8x1024xf32>
    %cst_198 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4486 = stablehlo.reduce(%4485 init: %cst_198) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4487 = stablehlo.broadcast_in_dim %4486, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4488 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4489 = stablehlo.divide %4487, %4488 : tensor<1x8x1xf32>
    %4490 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4491 = stablehlo.add %4489, %4490 : tensor<1x8x1xf32>
    %4492 = stablehlo.rsqrt %4491 : tensor<1x8x1xf32>
    %4493 = stablehlo.broadcast_in_dim %4492, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4494 = stablehlo.multiply %4484, %4493 : tensor<1x8x1024xf32>
    %4495 = stablehlo.convert %4494 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4496 = stablehlo.convert %arg291 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4497 = stablehlo.broadcast_in_dim %4496, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4498 = stablehlo.broadcast_in_dim %4497, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4499 = stablehlo.multiply %4498, %4495 : tensor<1x8x1024xbf16>
    %4500 = stablehlo.convert %arg289 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4501 = stablehlo.dot_general %4499, %4500, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4502 = call @silu(%4501) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4503 = stablehlo.convert %arg290 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4504 = stablehlo.dot_general %4499, %4503, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4505 = stablehlo.multiply %4502, %4504 : tensor<1x8x3072xbf16>
    %4506 = stablehlo.convert %arg288 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4507 = stablehlo.dot_general %4505, %4506, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4508 = stablehlo.add %4483, %4507 : tensor<1x8x1024xbf16>
    %4509 = stablehlo.convert %4508 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4510 = stablehlo.multiply %4509, %4509 : tensor<1x8x1024xf32>
    %cst_199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4511 = stablehlo.reduce(%4510 init: %cst_199) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4512 = stablehlo.broadcast_in_dim %4511, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4513 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4514 = stablehlo.divide %4512, %4513 : tensor<1x8x1xf32>
    %4515 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4516 = stablehlo.add %4514, %4515 : tensor<1x8x1xf32>
    %4517 = stablehlo.rsqrt %4516 : tensor<1x8x1xf32>
    %4518 = stablehlo.broadcast_in_dim %4517, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4519 = stablehlo.multiply %4509, %4518 : tensor<1x8x1024xf32>
    %4520 = stablehlo.convert %4519 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4521 = stablehlo.convert %arg298 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4522 = stablehlo.broadcast_in_dim %4521, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4523 = stablehlo.broadcast_in_dim %4522, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4524 = stablehlo.multiply %4523, %4520 : tensor<1x8x1024xbf16>
    %4525 = stablehlo.convert %arg307 : (tensor<1024x2048xf32>) -> tensor<1024x2048xbf16>
    %4526 = stablehlo.dot_general %4524, %4525, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x2048xbf16>) -> tensor<1x8x2048xbf16>
    %4527 = stablehlo.convert %arg304 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4528 = stablehlo.dot_general %4524, %4527, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4529 = stablehlo.convert %arg308 : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
    %4530 = stablehlo.dot_general %4524, %4529, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4531 = stablehlo.reshape %4526 : (tensor<1x8x2048xbf16>) -> tensor<1x8x16x128xbf16>
    %4532 = stablehlo.convert %4531 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x128xf32>
    %4533 = stablehlo.multiply %4532, %4532 : tensor<1x8x16x128xf32>
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4534 = stablehlo.reduce(%4533 init: %cst_200) applies stablehlo.add across dimensions = [3] : (tensor<1x8x16x128xf32>, tensor<f32>) -> tensor<1x8x16xf32>
    %4535 = stablehlo.broadcast_in_dim %4534, dims = [0, 1, 2] : (tensor<1x8x16xf32>) -> tensor<1x8x16x1xf32>
    %4536 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4537 = stablehlo.divide %4535, %4536 : tensor<1x8x16x1xf32>
    %4538 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x16x1xf32>
    %4539 = stablehlo.add %4537, %4538 : tensor<1x8x16x1xf32>
    %4540 = stablehlo.rsqrt %4539 : tensor<1x8x16x1xf32>
    %4541 = stablehlo.broadcast_in_dim %4540, dims = [0, 1, 2, 3] : (tensor<1x8x16x1xf32>) -> tensor<1x8x16x128xf32>
    %4542 = stablehlo.multiply %4532, %4541 : tensor<1x8x16x128xf32>
    %4543 = stablehlo.convert %4542 : (tensor<1x8x16x128xf32>) -> tensor<1x8x16x128xbf16>
    %4544 = stablehlo.convert %arg306 : (tensor<128xf32>) -> tensor<128xbf16>
    %4545 = stablehlo.broadcast_in_dim %4544, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4546 = stablehlo.broadcast_in_dim %4545, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4547 = stablehlo.multiply %4546, %4543 : tensor<1x8x16x128xbf16>
    %4548 = stablehlo.reshape %4528 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4549 = stablehlo.convert %4548 : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x128xf32>
    %4550 = stablehlo.multiply %4549, %4549 : tensor<1x8x8x128xf32>
    %cst_201 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4551 = stablehlo.reduce(%4550 init: %cst_201) applies stablehlo.add across dimensions = [3] : (tensor<1x8x8x128xf32>, tensor<f32>) -> tensor<1x8x8xf32>
    %4552 = stablehlo.broadcast_in_dim %4551, dims = [0, 1, 2] : (tensor<1x8x8xf32>) -> tensor<1x8x8x1xf32>
    %4553 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4554 = stablehlo.divide %4552, %4553 : tensor<1x8x8x1xf32>
    %4555 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x8x1xf32>
    %4556 = stablehlo.add %4554, %4555 : tensor<1x8x8x1xf32>
    %4557 = stablehlo.rsqrt %4556 : tensor<1x8x8x1xf32>
    %4558 = stablehlo.broadcast_in_dim %4557, dims = [0, 1, 2, 3] : (tensor<1x8x8x1xf32>) -> tensor<1x8x8x128xf32>
    %4559 = stablehlo.multiply %4549, %4558 : tensor<1x8x8x128xf32>
    %4560 = stablehlo.convert %4559 : (tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xbf16>
    %4561 = stablehlo.convert %arg303 : (tensor<128xf32>) -> tensor<128xbf16>
    %4562 = stablehlo.broadcast_in_dim %4561, dims = [3] : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %4563 = stablehlo.broadcast_in_dim %4562, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xbf16>) -> tensor<1x8x8x128xbf16>
    %4564 = stablehlo.multiply %4563, %4560 : tensor<1x8x8x128xbf16>
    %4565 = stablehlo.reshape %4530 : (tensor<1x8x1024xbf16>) -> tensor<1x8x8x128xbf16>
    %4566 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4567 = stablehlo.compare  LT, %7, %4566,  SIGNED : (tensor<1x8xi32>, tensor<1x8xi32>) -> tensor<1x8xi1>
    %4568 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<1x8xi32>
    %4569 = stablehlo.add %7, %4568 : tensor<1x8xi32>
    %4570 = stablehlo.select %4567, %4569, %7 : tensor<1x8xi1>, tensor<1x8xi32>
    %4571 = stablehlo.broadcast_in_dim %4570, dims = [0, 1] : (tensor<1x8xi32>) -> tensor<1x8x1xi32>
    %4572 = "stablehlo.gather"(%26, %4571) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<40960x128xf32>, tensor<1x8x1xi32>) -> tensor<1x8x128xf32>
    %4573 = stablehlo.slice %4572 [0:1, 0:8, 0:64] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4574 = stablehlo.slice %4572 [0:1, 0:8, 64:128] : (tensor<1x8x128xf32>) -> tensor<1x8x64xf32>
    %4575 = stablehlo.broadcast_in_dim %4573, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4576 = stablehlo.convert %4575 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4577 = stablehlo.broadcast_in_dim %4574, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4578 = stablehlo.convert %4577 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4579 = stablehlo.slice %4547 [0:1, 0:8, 0:16, 0:64] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4580 = stablehlo.slice %4547 [0:1, 0:8, 0:16, 64:128] : (tensor<1x8x16x128xbf16>) -> tensor<1x8x16x64xbf16>
    %4581 = stablehlo.broadcast_in_dim %4576, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4582 = stablehlo.multiply %4579, %4581 : tensor<1x8x16x64xbf16>
    %4583 = stablehlo.broadcast_in_dim %4578, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4584 = stablehlo.multiply %4580, %4583 : tensor<1x8x16x64xbf16>
    %4585 = stablehlo.subtract %4582, %4584 : tensor<1x8x16x64xbf16>
    %4586 = stablehlo.broadcast_in_dim %4576, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4587 = stablehlo.multiply %4580, %4586 : tensor<1x8x16x64xbf16>
    %4588 = stablehlo.broadcast_in_dim %4578, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x16x64xbf16>
    %4589 = stablehlo.multiply %4579, %4588 : tensor<1x8x16x64xbf16>
    %4590 = stablehlo.add %4587, %4589 : tensor<1x8x16x64xbf16>
    %4591 = stablehlo.concatenate %4585, %4590, dim = 3 : (tensor<1x8x16x64xbf16>, tensor<1x8x16x64xbf16>) -> tensor<1x8x16x128xbf16>
    %4592 = stablehlo.broadcast_in_dim %4573, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4593 = stablehlo.convert %4592 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4594 = stablehlo.broadcast_in_dim %4574, dims = [0, 1, 3] : (tensor<1x8x64xf32>) -> tensor<1x8x1x64xf32>
    %4595 = stablehlo.convert %4594 : (tensor<1x8x1x64xf32>) -> tensor<1x8x1x64xbf16>
    %4596 = stablehlo.slice %4564 [0:1, 0:8, 0:8, 0:64] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4597 = stablehlo.slice %4564 [0:1, 0:8, 0:8, 64:128] : (tensor<1x8x8x128xbf16>) -> tensor<1x8x8x64xbf16>
    %4598 = stablehlo.broadcast_in_dim %4593, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4599 = stablehlo.multiply %4596, %4598 : tensor<1x8x8x64xbf16>
    %4600 = stablehlo.broadcast_in_dim %4595, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4601 = stablehlo.multiply %4597, %4600 : tensor<1x8x8x64xbf16>
    %4602 = stablehlo.subtract %4599, %4601 : tensor<1x8x8x64xbf16>
    %4603 = stablehlo.broadcast_in_dim %4593, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4604 = stablehlo.multiply %4597, %4603 : tensor<1x8x8x64xbf16>
    %4605 = stablehlo.broadcast_in_dim %4595, dims = [0, 1, 2, 3] : (tensor<1x8x1x64xbf16>) -> tensor<1x8x8x64xbf16>
    %4606 = stablehlo.multiply %4596, %4605 : tensor<1x8x8x64xbf16>
    %4607 = stablehlo.add %4604, %4606 : tensor<1x8x8x64xbf16>
    %4608 = stablehlo.concatenate %4602, %4607, dim = 3 : (tensor<1x8x8x64xbf16>, tensor<1x8x8x64xbf16>) -> tensor<1x8x8x128xbf16>
    %4609 = stablehlo.broadcast_in_dim %3, dims = [0, 3] : (tensor<1x8xi1>) -> tensor<1x1x1x8xi1>
    %4610 = stablehlo.slice %25 [0:1, 0:1, 0:8, 0:8] : (tensor<1x1x128x128xi1>) -> tensor<1x1x8x8xi1>
    %4611 = stablehlo.broadcast_in_dim %4609, dims = [0, 1, 2, 3] : (tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1>
    %4612 = stablehlo.and %4611, %4610 : tensor<1x1x8x8xi1>
    %4613 = stablehlo.convert %4612 : (tensor<1x1x8x8xi1>) -> tensor<1x1x8x8xf32>
    %4614 = sdy.sharding_constraint %4591 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4615 = sdy.sharding_constraint %4608 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4616 = sdy.sharding_constraint %4565 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x8x128xbf16>
    %4617 = sdy.sharding_constraint %4613 <@mesh, [{}, {}, {}, {}]> : tensor<1x1x8x8xf32>
    %4618 = stablehlo.reshape %4614 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4619 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x8x8x2x128xbf16>
    %4620 = stablehlo.multiply %4618, %4619 : tensor<1x8x8x2x128xbf16>
    %4621 = stablehlo.dot_general %4615, %4620, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x8x2x128xbf16>) -> tensor<1x8x8x8x2xbf16>
    %4622 = stablehlo.transpose %4621, dims = [0, 1, 4, 3, 2] : (tensor<1x8x8x8x2xbf16>) -> tensor<1x8x2x8x8xbf16>
    %cst_202 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4623 = stablehlo.broadcast_in_dim %cst_202, dims = [] : (tensor<f32>) -> tensor<1x1x8x8xf32>
    %4624 = stablehlo.compare  NE, %4617, %4623,  FLOAT : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xi1>
    %4625 = stablehlo.convert %4624 : tensor<1x1x8x8xi1>
    %4626 = stablehlo.broadcast_in_dim %4625, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<1x8x8x8xi1>
    %4627 = stablehlo.reshape %4626 : (tensor<1x8x8x8xi1>) -> tensor<1x8x1x8x8xi1>
    %4628 = call @_where_83(%4627, %4622, %cst_12) : (tensor<1x8x1x8x8xi1>, tensor<1x8x2x8x8xbf16>, tensor<bf16>) -> tensor<1x8x2x8x8xbf16>
    %4629 = stablehlo.convert %4628 : (tensor<1x8x2x8x8xbf16>) -> tensor<1x8x2x8x8xf32>
    %cst_203 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4630 = stablehlo.reduce(%4629 init: %cst_203) applies stablehlo.maximum across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4631 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x8x2x8xf32>
    %4632 = stablehlo.maximum %4631, %4630 : tensor<1x8x2x8xf32>
    %4633 = stablehlo.broadcast_in_dim %4632, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4634 = stablehlo.broadcast_in_dim %4633, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4635 = stablehlo.subtract %4629, %4634 : tensor<1x8x2x8x8xf32>
    %4636 = stablehlo.exponential %4635 : tensor<1x8x2x8x8xf32>
    %cst_204 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4637 = stablehlo.reduce(%4636 init: %cst_204) applies stablehlo.add across dimensions = [4] : (tensor<1x8x2x8x8xf32>, tensor<f32>) -> tensor<1x8x2x8xf32>
    %4638 = stablehlo.broadcast_in_dim %4637, dims = [0, 1, 2, 3] : (tensor<1x8x2x8xf32>) -> tensor<1x8x2x8x1xf32>
    %4639 = stablehlo.broadcast_in_dim %4638, dims = [0, 1, 2, 3, 4] : (tensor<1x8x2x8x1xf32>) -> tensor<1x8x2x8x8xf32>
    %4640 = stablehlo.divide %4636, %4639 : tensor<1x8x2x8x8xf32>
    %4641 = stablehlo.convert %4640 : (tensor<1x8x2x8x8xf32>) -> tensor<1x8x2x8x8xbf16>
    %4642 = stablehlo.dot_general %4616, %4641, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [4], precision = [DEFAULT, DEFAULT] : (tensor<1x8x8x128xbf16>, tensor<1x8x2x8x8xbf16>) -> tensor<1x8x128x2x8xbf16>
    %4643 = stablehlo.transpose %4642, dims = [0, 4, 1, 3, 2] : (tensor<1x8x128x2x8xbf16>) -> tensor<1x8x8x2x128xbf16>
    %4644 = stablehlo.reshape %4643 : (tensor<1x8x8x2x128xbf16>) -> tensor<1x8x16x128xbf16>
    %4645 = sdy.sharding_constraint %4644 <@mesh, [{}, {}, {}, {}]> : tensor<1x8x16x128xbf16>
    %4646 = stablehlo.reshape %4645 : (tensor<1x8x16x128xbf16>) -> tensor<1x8x2048xbf16>
    %4647 = stablehlo.convert %arg305 : (tensor<2048x1024xf32>) -> tensor<2048x1024xbf16>
    %4648 = stablehlo.dot_general %4646, %4647, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4649 = stablehlo.add %4508, %4648 : tensor<1x8x1024xbf16>
    %4650 = stablehlo.convert %4649 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4651 = stablehlo.multiply %4650, %4650 : tensor<1x8x1024xf32>
    %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4652 = stablehlo.reduce(%4651 init: %cst_205) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4653 = stablehlo.broadcast_in_dim %4652, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4654 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4655 = stablehlo.divide %4653, %4654 : tensor<1x8x1xf32>
    %4656 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4657 = stablehlo.add %4655, %4656 : tensor<1x8x1xf32>
    %4658 = stablehlo.rsqrt %4657 : tensor<1x8x1xf32>
    %4659 = stablehlo.broadcast_in_dim %4658, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4660 = stablehlo.multiply %4650, %4659 : tensor<1x8x1024xf32>
    %4661 = stablehlo.convert %4660 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4662 = stablehlo.convert %arg302 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4663 = stablehlo.broadcast_in_dim %4662, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4664 = stablehlo.broadcast_in_dim %4663, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4665 = stablehlo.multiply %4664, %4661 : tensor<1x8x1024xbf16>
    %4666 = stablehlo.convert %arg300 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4667 = stablehlo.dot_general %4665, %4666, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4668 = call @silu(%4667) : (tensor<1x8x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4669 = stablehlo.convert %arg301 : (tensor<1024x3072xf32>) -> tensor<1024x3072xbf16>
    %4670 = stablehlo.dot_general %4665, %4669, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<1x8x3072xbf16>
    %4671 = stablehlo.multiply %4668, %4670 : tensor<1x8x3072xbf16>
    %4672 = stablehlo.convert %arg299 : (tensor<3072x1024xf32>) -> tensor<3072x1024xbf16>
    %4673 = stablehlo.dot_general %4671, %4672, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4674 = stablehlo.add %4649, %4673 : tensor<1x8x1024xbf16>
    %4675 = stablehlo.convert %4674 : (tensor<1x8x1024xbf16>) -> tensor<1x8x1024xf32>
    %4676 = stablehlo.multiply %4675, %4675 : tensor<1x8x1024xf32>
    %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4677 = stablehlo.reduce(%4676 init: %cst_206) applies stablehlo.add across dimensions = [2] : (tensor<1x8x1024xf32>, tensor<f32>) -> tensor<1x8xf32>
    %4678 = stablehlo.broadcast_in_dim %4677, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x1xf32>
    %4679 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4680 = stablehlo.divide %4678, %4679 : tensor<1x8x1xf32>
    %4681 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x8x1xf32>
    %4682 = stablehlo.add %4680, %4681 : tensor<1x8x1xf32>
    %4683 = stablehlo.rsqrt %4682 : tensor<1x8x1xf32>
    %4684 = stablehlo.broadcast_in_dim %4683, dims = [0, 1, 2] : (tensor<1x8x1xf32>) -> tensor<1x8x1024xf32>
    %4685 = stablehlo.multiply %4675, %4684 : tensor<1x8x1024xf32>
    %4686 = stablehlo.convert %4685 : (tensor<1x8x1024xf32>) -> tensor<1x8x1024xbf16>
    %4687 = stablehlo.convert %arg309 : (tensor<1024xf32>) -> tensor<1024xbf16>
    %4688 = stablehlo.broadcast_in_dim %4687, dims = [2] : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %4689 = stablehlo.broadcast_in_dim %4688, dims = [0, 1, 2] : (tensor<1x1x1024xbf16>) -> tensor<1x8x1024xbf16>
    %4690 = stablehlo.multiply %4689, %4686 : tensor<1x8x1024xbf16>
    %4691 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<151936x1024xf32>) -> tensor<1024x151936xf32>
    %4692 = stablehlo.convert %4691 : (tensor<1024x151936xf32>) -> tensor<1024x151936xbf16>
    %4693 = stablehlo.dot_general %4690, %4692, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x8x1024xbf16>, tensor<1024x151936xbf16>) -> tensor<1x8x151936xbf16>
    return %4693 : tensor<1x8x151936xbf16>
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
  func.func private @_where_83(%arg0: tensor<1x8x1x8x8xi1>, %arg1: tensor<1x8x2x8x8xbf16>, %arg2: tensor<bf16>) -> tensor<1x8x2x8x8xbf16> {
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
