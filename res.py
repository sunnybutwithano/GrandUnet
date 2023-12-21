import torch

res = """
[[tensor(0.1822), tensor(0.2990), tensor(0.2990), tensor(0.2990), tensor(0.3015), tensor(0.3467), tensor(0.3656), tensor(0.4183), tensor(0.5917), tensor(0.5766), tensor(0.6319), tensor(0.6118), tensor(0.6030), tensor(0.5025), tensor(0.4950), tensor(0.5678), tensor(0.6432), tensor(0.6043), tensor(0.7349), tensor(0.6621), tensor(0.7173), tensor(0.7575), tensor(0.6897), tensor(0.7236), tensor(0.7663), tensor(0.6985), tensor(0.7173), tensor(0.7575), tensor(0.7450), tensor(0.7462), tensor(0.7399), tensor(0.7688), tensor(0.7399), tensor(0.7638), tensor(0.7575), tensor(0.7814), tensor(0.7802), tensor(0.7776), tensor(0.7877), tensor(0.7739), tensor(0.7663), tensor(0.7714), tensor(0.7701), tensor(0.7789), tensor(0.7714), tensor(0.7714), tensor(0.7927), tensor(0.8015), tensor(0.7739), tensor(0.7802), tensor(0.8204), tensor(0.8229), tensor(0.7889), tensor(0.7889), tensor(0.7965), tensor(0.8279), tensor(0.7852), tensor(0.7952), tensor(0.8254), tensor(0.8128), tensor(0.8241), tensor(0.8153), tensor(0.8128), tensor(0.8103), tensor(0.8254), tensor(0.8543), tensor(0.8304), tensor(0.8103), tensor(0.8291), tensor(0.8204), tensor(0.8216), tensor(0.8166), tensor(0.8329), tensor(0.8442), tensor(0.8379), tensor(0.8229), tensor(0.8342), tensor(0.8505), tensor(0.8241), tensor(0.8128), tensor(0.8455), tensor(0.8593), tensor(0.8266), tensor(0.8191), tensor(0.8430), tensor(0.8467), tensor(0.8342), tensor(0.8216), tensor(0.8392), tensor(0.8354), tensor(0.8430), tensor(0.8291), tensor(0.8279), tensor(0.8317), tensor(0.8191), tensor(0.8229), tensor(0.8467), tensor(0.8467), tensor(0.8241), tensor(0.8266)], [tensor(0.1658), tensor(0.3028), tensor(0.3028), tensor(0.3028), tensor(0.3555), tensor(0.3769), tensor(0.4686), tensor(0.5151), tensor(0.5955), tensor(0.5779), tensor(0.5779), tensor(0.6106), tensor(0.6495), tensor(0.6168), tensor(0.6608), tensor(0.7085), tensor(0.7412), tensor(0.7525), tensor(0.7399), tensor(0.7676), tensor(0.7575), tensor(0.7601), tensor(0.7613), tensor(0.7538), tensor(0.7726), tensor(0.8028), tensor(0.7864), tensor(0.7965), tensor(0.8103), tensor(0.7764), tensor(0.8103), tensor(0.7877), tensor(0.7990), tensor(0.7990), tensor(0.7940), tensor(0.8053), tensor(0.8191), tensor(0.8141), tensor(0.7990), tensor(0.8090), tensor(0.8141), tensor(0.8065), tensor(0.8028), tensor(0.8015), tensor(0.8065), tensor(0.8153), tensor(0.7977), tensor(0.8078), tensor(0.8417), tensor(0.7864), tensor(0.8216), tensor(0.8128), tensor(0.8040), tensor(0.8141), tensor(0.8455), tensor(0.8153), tensor(0.7990), tensor(0.8204), tensor(0.8241), tensor(0.8405), tensor(0.8166), tensor(0.8103), tensor(0.8367), tensor(0.8241), tensor(0.8116), tensor(0.8291), tensor(0.8291), tensor(0.8204), tensor(0.8379), tensor(0.8430), tensor(0.8204), tensor(0.8254), tensor(0.8442), tensor(0.8405), tensor(0.8417), tensor(0.8329), tensor(0.8442), tensor(0.8455), tensor(0.8078), tensor(0.8317), tensor(0.8442), tensor(0.8317), tensor(0.8492), tensor(0.8342), tensor(0.8254), tensor(0.8342), tensor(0.8354), tensor(0.8317), tensor(0.8191), tensor(0.8354), tensor(0.8430), tensor(0.8505), tensor(0.8317), tensor(0.8291), tensor(0.8568), tensor(0.8430), tensor(0.8379), tensor(0.8329), tensor(0.8518), tensor(0.8392)], [tensor(0.2563), tensor(0.2802), tensor(0.2802), tensor(0.2802), tensor(0.2952), tensor(0.4736), tensor(0.4673), tensor(0.5653), tensor(0.5829), tensor(0.6608), tensor(0.6972), tensor(0.6859), tensor(0.7161), tensor(0.6043), tensor(0.5415), tensor(0.7362), tensor(0.7224), tensor(0.6520), tensor(0.7462), tensor(0.7098), tensor(0.7337), tensor(0.7500), tensor(0.7249), tensor(0.8141), tensor(0.7324), tensor(0.7563), tensor(0.6960), tensor(0.7802), tensor(0.7437), tensor(0.7224), tensor(0.7148), tensor(0.7789), tensor(0.7651), tensor(0.7613), tensor(0.7651), tensor(0.7852), tensor(0.7663), tensor(0.7852), tensor(0.7739), tensor(0.8053), tensor(0.8028), tensor(0.7676), tensor(0.7889), tensor(0.7902), tensor(0.7927), tensor(0.7915), tensor(0.7739), tensor(0.7990), tensor(0.7764), tensor(0.8053), tensor(0.7990), tensor(0.7701), tensor(0.7915), tensor(0.8003), tensor(0.7965), tensor(0.7726), tensor(0.8028), tensor(0.8141), tensor(0.7864), tensor(0.7814), tensor(0.8065), tensor(0.8153), tensor(0.7977), tensor(0.8053), tensor(0.8116), tensor(0.8254), tensor(0.8015), tensor(0.7751), tensor(0.8216), tensor(0.8216), tensor(0.7977), tensor(0.8103), tensor(0.8153), tensor(0.7977), tensor(0.8015), tensor(0.8128), tensor(0.7990), tensor(0.7990), tensor(0.7990), tensor(0.8003), tensor(0.8015), tensor(0.8216), tensor(0.8216), tensor(0.8003), tensor(0.8065), tensor(0.8078), tensor(0.8166), tensor(0.8065), tensor(0.7940), tensor(0.8053), tensor(0.8128), tensor(0.8153), tensor(0.8216), tensor(0.8116), tensor(0.7977), tensor(0.8178), tensor(0.8191), tensor(0.8229), tensor(0.8053), tensor(0.8090)], [tensor(0.3153), tensor(0.3153), tensor(0.3153), tensor(0.3178), tensor(0.3518), tensor(0.5013), tensor(0.5741), tensor(0.5854), tensor(0.5955), tensor(0.6131), tensor(0.5917), tensor(0.5955), tensor(0.6106), tensor(0.6947), tensor(0.7048), tensor(0.7462), tensor(0.6671), tensor(0.7198), tensor(0.7889), tensor(0.7814), tensor(0.7676), tensor(0.8015), tensor(0.8090), tensor(0.8028), tensor(0.7927), tensor(0.8078), tensor(0.8153), tensor(0.8090), tensor(0.7902), tensor(0.8241), tensor(0.8003), tensor(0.7852), tensor(0.8053), tensor(0.8229), tensor(0.8216), tensor(0.8291), tensor(0.8028), tensor(0.7952), tensor(0.8090), tensor(0.8128), tensor(0.8128), tensor(0.8103), tensor(0.8116), tensor(0.8040), tensor(0.7952), tensor(0.8254), tensor(0.8304), tensor(0.8204), tensor(0.8078), tensor(0.8090), tensor(0.8291), tensor(0.8003), tensor(0.8003), tensor(0.8053), tensor(0.8254), tensor(0.7889), tensor(0.8254), tensor(0.8367), tensor(0.8028), tensor(0.8141), tensor(0.8153), tensor(0.8405), tensor(0.8291), tensor(0.8191), tensor(0.8040), tensor(0.8241), tensor(0.8329), tensor(0.8354), tensor(0.8279), tensor(0.8090), tensor(0.8204), tensor(0.8342), tensor(0.8329), tensor(0.8103), tensor(0.8216), tensor(0.8103), tensor(0.8141), tensor(0.8367), tensor(0.8204), tensor(0.8241), tensor(0.8317), tensor(0.8279), tensor(0.8116), tensor(0.8178), tensor(0.8367), tensor(0.8442), tensor(0.8304), tensor(0.8229), tensor(0.8266), tensor(0.8317), tensor(0.8342), tensor(0.8291), tensor(0.8216), tensor(0.8229), tensor(0.8304), tensor(0.8053), tensor(0.8342), tensor(0.8216), tensor(0.8367), tensor(0.8204)], [tensor(0.3178), tensor(0.3103), tensor(0.3103), tensor(0.3103), tensor(0.3153), tensor(0.4422), tensor(0.4749), tensor(0.5013), tensor(0.5628), tensor(0.5616), tensor(0.6294), tensor(0.5917), tensor(0.7249), tensor(0.5214), tensor(0.6847), tensor(0.7073), tensor(0.6985), tensor(0.7324), tensor(0.7450), tensor(0.6947), tensor(0.7387), tensor(0.7261), tensor(0.7425), tensor(0.7324), tensor(0.7550), tensor(0.6859), tensor(0.7688), tensor(0.7613), tensor(0.7437), tensor(0.7827), tensor(0.7349), tensor(0.7764), tensor(0.7601), tensor(0.7525), tensor(0.7513), tensor(0.7889), tensor(0.7952), tensor(0.7575), tensor(0.7814), tensor(0.7927), tensor(0.7739), tensor(0.7952), tensor(0.7952), tensor(0.8015), tensor(0.8015), tensor(0.7965), tensor(0.7902), tensor(0.7814), tensor(0.8015), tensor(0.8015), tensor(0.7977), tensor(0.8053), tensor(0.7852), tensor(0.8078), tensor(0.8166), tensor(0.7940), tensor(0.7739), tensor(0.7927), tensor(0.7726), tensor(0.7977), tensor(0.7927), tensor(0.8116), tensor(0.8078), tensor(0.8053), tensor(0.8028), tensor(0.8028), tensor(0.8241), tensor(0.8254), tensor(0.8065), tensor(0.8128), tensor(0.8040), tensor(0.8028), tensor(0.8116), tensor(0.8329), tensor(0.8229), tensor(0.8178), tensor(0.8153), tensor(0.8166), tensor(0.8204), tensor(0.8216), tensor(0.8141), tensor(0.8078), tensor(0.8141), tensor(0.8103), tensor(0.8241), tensor(0.8191), tensor(0.8204), tensor(0.8141), tensor(0.8191), tensor(0.8291), tensor(0.8216), tensor(0.8254), tensor(0.8241), tensor(0.8254), tensor(0.8166), tensor(0.8241), tensor(0.8153), tensor(0.8116), tensor(0.8279), tensor(0.8342)], [tensor(0.2651), tensor(0.2651), tensor(0.2651), tensor(0.2651), tensor(0.3392), tensor(0.4158), tensor(0.4686), tensor(0.5113), tensor(0.5653), tensor(0.6696), tensor(0.6859), tensor(0.6771), tensor(0.7286), tensor(0.6508), tensor(0.6872), tensor(0.7286), tensor(0.6947), tensor(0.7525), tensor(0.7173), tensor(0.7651), tensor(0.7663), tensor(0.7588), tensor(0.7563), tensor(0.7161), tensor(0.7450), tensor(0.7877), tensor(0.7952), tensor(0.7500), tensor(0.7877), tensor(0.7701), tensor(0.7764), tensor(0.7776), tensor(0.7977), tensor(0.7789), tensor(0.8003), tensor(0.7927), tensor(0.7802), tensor(0.8003), tensor(0.8028), tensor(0.7852), tensor(0.7902), tensor(0.8229), tensor(0.8003), tensor(0.8065), tensor(0.8304), tensor(0.8065), tensor(0.7990), tensor(0.8166), tensor(0.8216), tensor(0.8204), tensor(0.8229), tensor(0.8090), tensor(0.8241), tensor(0.8354), tensor(0.8329), tensor(0.8191), tensor(0.8254), tensor(0.8442), tensor(0.8128), tensor(0.8279), tensor(0.8241), tensor(0.8128), tensor(0.8116), tensor(0.8342), tensor(0.8304), tensor(0.8153), tensor(0.8191), tensor(0.8153), tensor(0.8442), tensor(0.8367), tensor(0.8254), tensor(0.8329), tensor(0.8379), tensor(0.8291), tensor(0.8304), tensor(0.8417), tensor(0.8279), tensor(0.8166), tensor(0.8405), tensor(0.8367), tensor(0.8153), tensor(0.8254), tensor(0.8379), tensor(0.8379), tensor(0.8317), tensor(0.8405), tensor(0.8430), tensor(0.8266), tensor(0.8379), tensor(0.8455), tensor(0.8405), tensor(0.8367), tensor(0.8455), tensor(0.8430), tensor(0.8417), tensor(0.8241), tensor(0.8291), tensor(0.8480), tensor(0.8103), tensor(0.8065)], [tensor(0.1985), tensor(0.2864), tensor(0.2852), tensor(0.2889), tensor(0.3342), tensor(0.3580), tensor(0.4912), tensor(0.5766), tensor(0.6972), tensor(0.6407), tensor(0.6859), tensor(0.6771), tensor(0.7161), tensor(0.5389), tensor(0.7286), tensor(0.7148), tensor(0.7714), tensor(0.7676), tensor(0.7877), tensor(0.7299), tensor(0.7387), tensor(0.7940), tensor(0.7136), tensor(0.8015), tensor(0.7877), tensor(0.7789), tensor(0.7889), tensor(0.7952), tensor(0.7726), tensor(0.8015), tensor(0.7299), tensor(0.7990), tensor(0.7952), tensor(0.7877), tensor(0.7789), tensor(0.7915), tensor(0.7827), tensor(0.8015), tensor(0.7701), tensor(0.8065), tensor(0.7977), tensor(0.7902), tensor(0.8040), tensor(0.7965), tensor(0.8116), tensor(0.8141), tensor(0.7814), tensor(0.8166), tensor(0.8153), tensor(0.8040), tensor(0.8166), tensor(0.8015), tensor(0.8266), tensor(0.8178), tensor(0.8128), tensor(0.8204), tensor(0.8141), tensor(0.8128), tensor(0.8216), tensor(0.8204), tensor(0.8266), tensor(0.8204), tensor(0.8229), tensor(0.8266), tensor(0.8166), tensor(0.8153), tensor(0.8254), tensor(0.8266), tensor(0.8204), tensor(0.8254), tensor(0.8266), tensor(0.8304), tensor(0.8329), tensor(0.8329), tensor(0.8317), tensor(0.8304), tensor(0.8304), tensor(0.8317), tensor(0.8266), tensor(0.8241), tensor(0.8317), tensor(0.8266), tensor(0.8266), tensor(0.8342), tensor(0.8367), tensor(0.8279), tensor(0.8291), tensor(0.8291), tensor(0.8317), tensor(0.8317), tensor(0.8342), tensor(0.8392), tensor(0.8417), tensor(0.8367), tensor(0.8392), tensor(0.8291), tensor(0.8329), tensor(0.8329), tensor(0.8266), tensor(0.8254)], [tensor(0.1746), tensor(0.1947), tensor(0.3128), tensor(0.3957), tensor(0.3141), tensor(0.3153), tensor(0.4472), tensor(0.5113), tensor(0.5930), tensor(0.6005), tensor(0.6420), tensor(0.6922), tensor(0.6658), tensor(0.6457), tensor(0.6093), tensor(0.7337), tensor(0.6193), tensor(0.7136), tensor(0.7500), tensor(0.7023), tensor(0.7412), tensor(0.7437), tensor(0.7651), tensor(0.8015), tensor(0.7337), tensor(0.7575), tensor(0.7663), tensor(0.7940), tensor(0.7425), tensor(0.8053), tensor(0.7764), tensor(0.7776), tensor(0.7751), tensor(0.7714), tensor(0.7676), tensor(0.8015), tensor(0.7676), tensor(0.8040), tensor(0.7952), tensor(0.7864), tensor(0.7588), tensor(0.8090), tensor(0.7714), tensor(0.7927), tensor(0.8266), tensor(0.8040), tensor(0.7789), tensor(0.8015), tensor(0.8266), tensor(0.7764), tensor(0.8304), tensor(0.8040), tensor(0.7877), tensor(0.8204), tensor(0.7940), tensor(0.8241), tensor(0.8342), tensor(0.8178), tensor(0.8304), tensor(0.8128), tensor(0.7751), tensor(0.8229), tensor(0.8229), tensor(0.8040), tensor(0.7990), tensor(0.8128), tensor(0.8191), tensor(0.8392), tensor(0.8128), tensor(0.8379), tensor(0.8229), tensor(0.8191), tensor(0.8367), tensor(0.8329), tensor(0.8367), tensor(0.8078), tensor(0.8266), tensor(0.8304), tensor(0.8128), tensor(0.8342), tensor(0.8455), tensor(0.8053), tensor(0.8241), tensor(0.8455), tensor(0.8417), tensor(0.8304), tensor(0.8204), tensor(0.8354), tensor(0.8304), tensor(0.8153), tensor(0.8053), tensor(0.8254), tensor(0.8405), tensor(0.8103), tensor(0.8178), tensor(0.8442), tensor(0.8266), tensor(0.8354), tensor(0.8254), tensor(0.8342)], [tensor(0.3003), tensor(0.3003), tensor(0.3003), tensor(0.3003), tensor(0.3467), tensor(0.5126), tensor(0.5214), tensor(0.5515), tensor(0.6482), tensor(0.6332), tensor(0.6219), tensor(0.7098), tensor(0.5955), tensor(0.6709), tensor(0.6369), tensor(0.6558), tensor(0.6420), tensor(0.7638), tensor(0.6470), tensor(0.6608), tensor(0.7500), tensor(0.6796), tensor(0.6709), tensor(0.7060), tensor(0.7726), tensor(0.7148), tensor(0.7286), tensor(0.7085), tensor(0.7073), tensor(0.7374), tensor(0.7136), tensor(0.7676), tensor(0.6935), tensor(0.6759), tensor(0.7902), tensor(0.7487), tensor(0.7814), tensor(0.7487), tensor(0.7286), tensor(0.7500), tensor(0.7462), tensor(0.7651), tensor(0.7412), tensor(0.7701), tensor(0.7563), tensor(0.7726), tensor(0.8065), tensor(0.7450), tensor(0.7399), tensor(0.7977), tensor(0.7827), tensor(0.8065), tensor(0.7877), tensor(0.7902), tensor(0.8153), tensor(0.8116), tensor(0.7940), tensor(0.7714), tensor(0.7701), tensor(0.7764), tensor(0.7814), tensor(0.7990), tensor(0.7977), tensor(0.8003), tensor(0.8040), tensor(0.7940), tensor(0.7839), tensor(0.8028), tensor(0.7940), tensor(0.7714), tensor(0.8103), tensor(0.7751), tensor(0.8053), tensor(0.8204), tensor(0.7952), tensor(0.8065), tensor(0.8178), tensor(0.8053), tensor(0.8053), tensor(0.8065), tensor(0.8078), tensor(0.8103), tensor(0.7965), tensor(0.7751), tensor(0.8166), tensor(0.8191), tensor(0.7927), tensor(0.8166), tensor(0.8141), tensor(0.8015), tensor(0.7990), tensor(0.8078), tensor(0.8141), tensor(0.7965), tensor(0.8153), tensor(0.8254), tensor(0.8090), tensor(0.8254), tensor(0.8128), tensor(0.8116)], [tensor(0.3015), tensor(0.3015), tensor(0.3015), tensor(0.3392), tensor(0.3819), tensor(0.4384), tensor(0.5163), tensor(0.5980), tensor(0.6030), tensor(0.5867), tensor(0.7136), tensor(0.4209), tensor(0.5540), tensor(0.6457), tensor(0.6080), tensor(0.6420), tensor(0.6796), tensor(0.7035), tensor(0.7500), tensor(0.7236), tensor(0.7802), tensor(0.7688), tensor(0.7915), tensor(0.7638), tensor(0.7802), tensor(0.7877), tensor(0.8015), tensor(0.7864), tensor(0.7852), tensor(0.7990), tensor(0.7864), tensor(0.7538), tensor(0.7927), tensor(0.7802), tensor(0.7965), tensor(0.7789), tensor(0.7676), tensor(0.7902), tensor(0.8153), tensor(0.7538), tensor(0.8015), tensor(0.7977), tensor(0.7889), tensor(0.7864), tensor(0.7776), tensor(0.8065), tensor(0.8065), tensor(0.8153), tensor(0.8090), tensor(0.7952), tensor(0.8028), tensor(0.7814), tensor(0.8178), tensor(0.8229), tensor(0.7977), tensor(0.8040), tensor(0.8216), tensor(0.8304), tensor(0.8191), tensor(0.8204), tensor(0.8028), tensor(0.8304), tensor(0.7802), tensor(0.8178), tensor(0.8090), tensor(0.8090), tensor(0.8166), tensor(0.7990), tensor(0.8204), tensor(0.8455), tensor(0.8254), tensor(0.8342), tensor(0.8342), tensor(0.8191), tensor(0.8254), tensor(0.8166), tensor(0.8304), tensor(0.8442), tensor(0.8317), tensor(0.8317), tensor(0.8417), tensor(0.8480), tensor(0.8254), tensor(0.8153), tensor(0.8430), tensor(0.8028), tensor(0.8291), tensor(0.8442), tensor(0.8329), tensor(0.8229), tensor(0.8430), tensor(0.8342), tensor(0.8367), tensor(0.8329), tensor(0.8279), tensor(0.8116), tensor(0.8291), tensor(0.8467), tensor(0.8216), tensor(0.8480)]]

[[tensor(0.1771), tensor(0.2777), tensor(0.2777), tensor(0.2777), tensor(0.2797), tensor(0.3340), tensor(0.3481), tensor(0.4286), tensor(0.5855), tensor(0.5634), tensor(0.6358), tensor(0.5976), tensor(0.6016), tensor(0.5010), tensor(0.4748), tensor(0.5372), tensor(0.6398), tensor(0.5755), tensor(0.7022), tensor(0.6559), tensor(0.6962), tensor(0.7425), tensor(0.6700), tensor(0.7203), tensor(0.7807), tensor(0.6861), tensor(0.6982), tensor(0.7485), tensor(0.7324), tensor(0.7324), tensor(0.7445), tensor(0.7827), tensor(0.7324), tensor(0.7525), tensor(0.7545), tensor(0.7928), tensor(0.7887), tensor(0.7767), tensor(0.7968), tensor(0.7847), tensor(0.7746), tensor(0.7847), tensor(0.7887), tensor(0.7827), tensor(0.7847), tensor(0.7948), tensor(0.7988), tensor(0.8149), tensor(0.7948), tensor(0.7907), tensor(0.8149), tensor(0.8129), tensor(0.8028), tensor(0.7928), tensor(0.7907), tensor(0.8109), tensor(0.7887), tensor(0.7968), tensor(0.8189), tensor(0.7988), tensor(0.8209), tensor(0.8169), tensor(0.7988), tensor(0.8129), tensor(0.8149), tensor(0.8350), tensor(0.8169), tensor(0.8089), tensor(0.8089), tensor(0.8229), tensor(0.8129), tensor(0.8008), tensor(0.8270), tensor(0.8249), tensor(0.8109), tensor(0.8068), tensor(0.8189), tensor(0.8330), tensor(0.8089), tensor(0.7968), tensor(0.8270), tensor(0.8270), tensor(0.7988), tensor(0.8149), tensor(0.8229), tensor(0.8270), tensor(0.8169), tensor(0.8028), tensor(0.8149), tensor(0.8229), tensor(0.8169), tensor(0.8149), tensor(0.8229), tensor(0.8109), tensor(0.8089), tensor(0.8109), tensor(0.8229), tensor(0.8229), tensor(0.8028), tensor(0.8089)], [tensor(0.1670), tensor(0.2958), tensor(0.2958), tensor(0.2958), tensor(0.3380), tensor(0.3783), tensor(0.4648), tensor(0.5111), tensor(0.6076), tensor(0.5634), tensor(0.5815), tensor(0.5996), tensor(0.6439), tensor(0.6076), tensor(0.6499), tensor(0.7163), tensor(0.7586), tensor(0.7525), tensor(0.7726), tensor(0.7887), tensor(0.7767), tensor(0.7907), tensor(0.7928), tensor(0.7485), tensor(0.8089), tensor(0.8048), tensor(0.7807), tensor(0.8129), tensor(0.8209), tensor(0.7626), tensor(0.8209), tensor(0.8048), tensor(0.8229), tensor(0.8109), tensor(0.8129), tensor(0.8270), tensor(0.8129), tensor(0.8270), tensor(0.8169), tensor(0.8290), tensor(0.7988), tensor(0.8229), tensor(0.8330), tensor(0.8028), tensor(0.8209), tensor(0.8209), tensor(0.8008), tensor(0.8109), tensor(0.8330), tensor(0.8068), tensor(0.8109), tensor(0.8189), tensor(0.8270), tensor(0.7867), tensor(0.8370), tensor(0.8350), tensor(0.8048), tensor(0.8089), tensor(0.8169), tensor(0.8229), tensor(0.8270), tensor(0.8008), tensor(0.8431), tensor(0.8410), tensor(0.8028), tensor(0.8330), tensor(0.8068), tensor(0.8189), tensor(0.8471), tensor(0.8571), tensor(0.8249), tensor(0.8310), tensor(0.8451), tensor(0.8390), tensor(0.8491), tensor(0.8330), tensor(0.8189), tensor(0.8270), tensor(0.8089), tensor(0.8370), tensor(0.8370), tensor(0.8370), tensor(0.8511), tensor(0.8290), tensor(0.8270), tensor(0.8229), tensor(0.8531), tensor(0.8350), tensor(0.8109), tensor(0.8330), tensor(0.8491), tensor(0.8330), tensor(0.8209), tensor(0.8249), tensor(0.8471), tensor(0.8410), tensor(0.8390), tensor(0.8189), tensor(0.8390), tensor(0.8390)], [tensor(0.2696), tensor(0.3038), tensor(0.3038), tensor(0.3038), tensor(0.3239), tensor(0.5131), tensor(0.5070), tensor(0.5956), tensor(0.6056), tensor(0.6901), tensor(0.7384), tensor(0.7324), tensor(0.7082), tensor(0.6419), tensor(0.5312), tensor(0.7726), tensor(0.7284), tensor(0.6660), tensor(0.7767), tensor(0.7223), tensor(0.7143), tensor(0.7887), tensor(0.7565), tensor(0.8209), tensor(0.7887), tensor(0.7626), tensor(0.7404), tensor(0.7968), tensor(0.7545), tensor(0.7606), tensor(0.7082), tensor(0.8189), tensor(0.7968), tensor(0.7807), tensor(0.8008), tensor(0.7907), tensor(0.7807), tensor(0.7928), tensor(0.7746), tensor(0.8270), tensor(0.8048), tensor(0.7666), tensor(0.8008), tensor(0.8089), tensor(0.8068), tensor(0.7948), tensor(0.7988), tensor(0.8209), tensor(0.7867), tensor(0.8209), tensor(0.8169), tensor(0.7646), tensor(0.7907), tensor(0.8209), tensor(0.8129), tensor(0.7686), tensor(0.8390), tensor(0.8189), tensor(0.8149), tensor(0.8048), tensor(0.8169), tensor(0.8330), tensor(0.8149), tensor(0.8209), tensor(0.8209), tensor(0.8350), tensor(0.8270), tensor(0.8028), tensor(0.8209), tensor(0.8350), tensor(0.8189), tensor(0.8290), tensor(0.8431), tensor(0.8290), tensor(0.8209), tensor(0.8109), tensor(0.8270), tensor(0.8330), tensor(0.8290), tensor(0.8209), tensor(0.8249), tensor(0.8531), tensor(0.8451), tensor(0.8310), tensor(0.8350), tensor(0.8028), tensor(0.8511), tensor(0.8229), tensor(0.8068), tensor(0.8229), tensor(0.8451), tensor(0.8390), tensor(0.8330), tensor(0.8471), tensor(0.8209), tensor(0.8390), tensor(0.8431), tensor(0.8471), tensor(0.8431), tensor(0.8249)], [tensor(0.2575), tensor(0.2575), tensor(0.2575), tensor(0.2656), tensor(0.3219), tensor(0.5030), tensor(0.5835), tensor(0.5996), tensor(0.6137), tensor(0.6499), tensor(0.6197), tensor(0.6398), tensor(0.6278), tensor(0.7002), tensor(0.7445), tensor(0.7606), tensor(0.7143), tensor(0.7364), tensor(0.8270), tensor(0.8089), tensor(0.7847), tensor(0.8068), tensor(0.8169), tensor(0.8310), tensor(0.8229), tensor(0.8008), tensor(0.8310), tensor(0.8169), tensor(0.8189), tensor(0.8370), tensor(0.8229), tensor(0.7968), tensor(0.8290), tensor(0.8330), tensor(0.8290), tensor(0.8451), tensor(0.8169), tensor(0.8270), tensor(0.8350), tensor(0.8209), tensor(0.8169), tensor(0.8310), tensor(0.8209), tensor(0.8310), tensor(0.8310), tensor(0.8169), tensor(0.8390), tensor(0.8370), tensor(0.8129), tensor(0.8390), tensor(0.8370), tensor(0.8189), tensor(0.8229), tensor(0.8310), tensor(0.8310), tensor(0.8008), tensor(0.8370), tensor(0.8431), tensor(0.8249), tensor(0.8249), tensor(0.8290), tensor(0.8410), tensor(0.8310), tensor(0.8270), tensor(0.8270), tensor(0.8390), tensor(0.8431), tensor(0.8471), tensor(0.8410), tensor(0.8330), tensor(0.8270), tensor(0.8370), tensor(0.8370), tensor(0.8209), tensor(0.8370), tensor(0.8370), tensor(0.8330), tensor(0.8249), tensor(0.8310), tensor(0.8451), tensor(0.8410), tensor(0.8330), tensor(0.8209), tensor(0.8189), tensor(0.8390), tensor(0.8451), tensor(0.8350), tensor(0.8290), tensor(0.8330), tensor(0.8390), tensor(0.8390), tensor(0.8330), tensor(0.8310), tensor(0.8310), tensor(0.8249), tensor(0.8149), tensor(0.8390), tensor(0.8249), tensor(0.8390), tensor(0.8370)], [tensor(0.2897), tensor(0.2837), tensor(0.2837), tensor(0.2837), tensor(0.2857), tensor(0.4145), tensor(0.4588), tensor(0.4970), tensor(0.5433), tensor(0.5352), tensor(0.6157), tensor(0.5634), tensor(0.7344), tensor(0.4970), tensor(0.6720), tensor(0.7042), tensor(0.6881), tensor(0.7425), tensor(0.7606), tensor(0.6922), tensor(0.7465), tensor(0.7203), tensor(0.7646), tensor(0.7404), tensor(0.7767), tensor(0.6881), tensor(0.7706), tensor(0.7726), tensor(0.7344), tensor(0.7887), tensor(0.7203), tensor(0.7867), tensor(0.7525), tensor(0.7646), tensor(0.7525), tensor(0.8089), tensor(0.8068), tensor(0.7807), tensor(0.7847), tensor(0.8129), tensor(0.7907), tensor(0.8089), tensor(0.8068), tensor(0.8129), tensor(0.8189), tensor(0.8028), tensor(0.8129), tensor(0.8068), tensor(0.8028), tensor(0.8189), tensor(0.8089), tensor(0.8109), tensor(0.7948), tensor(0.8169), tensor(0.8290), tensor(0.8189), tensor(0.7787), tensor(0.7988), tensor(0.7867), tensor(0.8109), tensor(0.8089), tensor(0.8229), tensor(0.8149), tensor(0.8068), tensor(0.7988), tensor(0.8008), tensor(0.8310), tensor(0.8290), tensor(0.8048), tensor(0.8129), tensor(0.8209), tensor(0.8109), tensor(0.8089), tensor(0.8189), tensor(0.8270), tensor(0.8129), tensor(0.8109), tensor(0.8370), tensor(0.8209), tensor(0.8209), tensor(0.8149), tensor(0.8048), tensor(0.8249), tensor(0.8109), tensor(0.8209), tensor(0.8129), tensor(0.8310), tensor(0.8209), tensor(0.8209), tensor(0.8209), tensor(0.8109), tensor(0.8330), tensor(0.8330), tensor(0.8249), tensor(0.8149), tensor(0.8330), tensor(0.8129), tensor(0.8089), tensor(0.8249), tensor(0.8370)], [tensor(0.3018), tensor(0.3018), tensor(0.3018), tensor(0.3018), tensor(0.3400), tensor(0.4004), tensor(0.4567), tensor(0.5050), tensor(0.5634), tensor(0.6338), tensor(0.6660), tensor(0.6439), tensor(0.7002), tensor(0.6258), tensor(0.6740), tensor(0.7022), tensor(0.6781), tensor(0.7384), tensor(0.7264), tensor(0.7485), tensor(0.7746), tensor(0.7686), tensor(0.7626), tensor(0.7304), tensor(0.7364), tensor(0.8028), tensor(0.7867), tensor(0.7686), tensor(0.7887), tensor(0.7767), tensor(0.7787), tensor(0.7787), tensor(0.7968), tensor(0.7948), tensor(0.8169), tensor(0.7746), tensor(0.7928), tensor(0.8129), tensor(0.7928), tensor(0.7928), tensor(0.7928), tensor(0.8068), tensor(0.8169), tensor(0.8149), tensor(0.8249), tensor(0.8169), tensor(0.7928), tensor(0.8028), tensor(0.8189), tensor(0.8169), tensor(0.8149), tensor(0.8008), tensor(0.8169), tensor(0.8290), tensor(0.8249), tensor(0.8189), tensor(0.8229), tensor(0.8229), tensor(0.8109), tensor(0.8169), tensor(0.8370), tensor(0.8089), tensor(0.8149), tensor(0.8330), tensor(0.8229), tensor(0.8068), tensor(0.8270), tensor(0.8089), tensor(0.8330), tensor(0.8410), tensor(0.8229), tensor(0.8270), tensor(0.8370), tensor(0.8410), tensor(0.8270), tensor(0.8511), tensor(0.8370), tensor(0.8048), tensor(0.8310), tensor(0.8249), tensor(0.8149), tensor(0.8149), tensor(0.8410), tensor(0.8350), tensor(0.8350), tensor(0.8511), tensor(0.8431), tensor(0.8270), tensor(0.8350), tensor(0.8511), tensor(0.8330), tensor(0.8310), tensor(0.8431), tensor(0.8531), tensor(0.8370), tensor(0.8169), tensor(0.8270), tensor(0.8551), tensor(0.8129), tensor(0.8048)], [tensor(0.1610), tensor(0.3018), tensor(0.3018), tensor(0.3038), tensor(0.3581), tensor(0.3843), tensor(0.4950), tensor(0.5533), tensor(0.7082), tensor(0.6217), tensor(0.6801), tensor(0.6740), tensor(0.7243), tensor(0.5734), tensor(0.7183), tensor(0.7223), tensor(0.7706), tensor(0.7606), tensor(0.7928), tensor(0.7485), tensor(0.7264), tensor(0.7887), tensor(0.7284), tensor(0.8089), tensor(0.7807), tensor(0.7787), tensor(0.7827), tensor(0.7807), tensor(0.7586), tensor(0.7948), tensor(0.7203), tensor(0.7867), tensor(0.7847), tensor(0.7867), tensor(0.7646), tensor(0.7787), tensor(0.7948), tensor(0.7928), tensor(0.7626), tensor(0.7767), tensor(0.7887), tensor(0.7928), tensor(0.7948), tensor(0.7928), tensor(0.8109), tensor(0.8089), tensor(0.7787), tensor(0.8068), tensor(0.8068), tensor(0.7928), tensor(0.8048), tensor(0.7928), tensor(0.8068), tensor(0.8068), tensor(0.7988), tensor(0.8089), tensor(0.8008), tensor(0.8048), tensor(0.8048), tensor(0.8008), tensor(0.8028), tensor(0.7948), tensor(0.8109), tensor(0.8068), tensor(0.7948), tensor(0.8028), tensor(0.8028), tensor(0.7968), tensor(0.8028), tensor(0.7968), tensor(0.7948), tensor(0.8109), tensor(0.8169), tensor(0.8068), tensor(0.7988), tensor(0.8089), tensor(0.8129), tensor(0.8048), tensor(0.7948), tensor(0.8089), tensor(0.8008), tensor(0.8068), tensor(0.8089), tensor(0.8028), tensor(0.8008), tensor(0.8028), tensor(0.8109), tensor(0.7948), tensor(0.8109), tensor(0.8109), tensor(0.8048), tensor(0.8189), tensor(0.8189), tensor(0.8129), tensor(0.8089), tensor(0.8008), tensor(0.8028), tensor(0.8089), tensor(0.8008), tensor(0.7988)], [tensor(0.1449), tensor(0.1730), tensor(0.3078), tensor(0.3903), tensor(0.3421), tensor(0.3441), tensor(0.4406), tensor(0.4970), tensor(0.5936), tensor(0.5956), tensor(0.6479), tensor(0.6922), tensor(0.6700), tensor(0.6620), tensor(0.6479), tensor(0.7344), tensor(0.6439), tensor(0.7183), tensor(0.7767), tensor(0.7304), tensor(0.7364), tensor(0.7545), tensor(0.7907), tensor(0.8109), tensor(0.7284), tensor(0.7746), tensor(0.7606), tensor(0.8089), tensor(0.7545), tensor(0.8169), tensor(0.7907), tensor(0.7726), tensor(0.7787), tensor(0.7746), tensor(0.7706), tensor(0.8008), tensor(0.7787), tensor(0.7887), tensor(0.7706), tensor(0.7928), tensor(0.7425), tensor(0.8129), tensor(0.7767), tensor(0.7787), tensor(0.8451), tensor(0.8109), tensor(0.7767), tensor(0.8028), tensor(0.8290), tensor(0.7666), tensor(0.8209), tensor(0.7928), tensor(0.7807), tensor(0.8169), tensor(0.7867), tensor(0.8089), tensor(0.8310), tensor(0.8129), tensor(0.8330), tensor(0.8089), tensor(0.7606), tensor(0.8350), tensor(0.8068), tensor(0.7907), tensor(0.8008), tensor(0.8169), tensor(0.8209), tensor(0.8431), tensor(0.8089), tensor(0.8350), tensor(0.8149), tensor(0.8109), tensor(0.8390), tensor(0.8491), tensor(0.8270), tensor(0.8089), tensor(0.8149), tensor(0.8410), tensor(0.8169), tensor(0.8330), tensor(0.8370), tensor(0.7928), tensor(0.8229), tensor(0.8431), tensor(0.8431), tensor(0.8229), tensor(0.8209), tensor(0.8350), tensor(0.8330), tensor(0.8109), tensor(0.7948), tensor(0.8310), tensor(0.8511), tensor(0.7968), tensor(0.8149), tensor(0.8410), tensor(0.8229), tensor(0.8370), tensor(0.8350), tensor(0.8249)], [tensor(0.2475), tensor(0.2475), tensor(0.2475), tensor(0.2475), tensor(0.2897), tensor(0.4608), tensor(0.4688), tensor(0.5030), tensor(0.6016), tensor(0.5895), tensor(0.5714), tensor(0.6962), tensor(0.5412), tensor(0.6559), tensor(0.5895), tensor(0.6237), tensor(0.5855), tensor(0.7384), tensor(0.6117), tensor(0.6419), tensor(0.7143), tensor(0.6479), tensor(0.6539), tensor(0.6962), tensor(0.7445), tensor(0.6700), tensor(0.7143), tensor(0.6942), tensor(0.6861), tensor(0.7143), tensor(0.6499), tensor(0.7646), tensor(0.6841), tensor(0.6459), tensor(0.7887), tensor(0.7082), tensor(0.7404), tensor(0.7445), tensor(0.7082), tensor(0.7284), tensor(0.7485), tensor(0.7425), tensor(0.7264), tensor(0.7445), tensor(0.7384), tensor(0.7505), tensor(0.7807), tensor(0.7143), tensor(0.7425), tensor(0.7606), tensor(0.7626), tensor(0.7867), tensor(0.7666), tensor(0.7767), tensor(0.7928), tensor(0.7928), tensor(0.7686), tensor(0.7364), tensor(0.7746), tensor(0.7626), tensor(0.7767), tensor(0.7948), tensor(0.7807), tensor(0.7787), tensor(0.7887), tensor(0.7787), tensor(0.7626), tensor(0.7867), tensor(0.7726), tensor(0.7565), tensor(0.7968), tensor(0.7525), tensor(0.7928), tensor(0.7968), tensor(0.7706), tensor(0.7827), tensor(0.7968), tensor(0.7968), tensor(0.7948), tensor(0.7968), tensor(0.7928), tensor(0.7887), tensor(0.7787), tensor(0.7606), tensor(0.8129), tensor(0.8068), tensor(0.7706), tensor(0.7968), tensor(0.8008), tensor(0.7867), tensor(0.7847), tensor(0.8109), tensor(0.7968), tensor(0.7787), tensor(0.8028), tensor(0.8229), tensor(0.7867), tensor(0.8028), tensor(0.7968), tensor(0.7968)], [tensor(0.3099), tensor(0.3099), tensor(0.3099), tensor(0.3481), tensor(0.3843), tensor(0.4386), tensor(0.5392), tensor(0.5895), tensor(0.6217), tensor(0.5654), tensor(0.7143), tensor(0.3883), tensor(0.5252), tensor(0.6378), tensor(0.5714), tensor(0.6358), tensor(0.6740), tensor(0.6901), tensor(0.7324), tensor(0.7264), tensor(0.7887), tensor(0.7606), tensor(0.7767), tensor(0.7787), tensor(0.7666), tensor(0.7666), tensor(0.7867), tensor(0.7746), tensor(0.7827), tensor(0.7847), tensor(0.7726), tensor(0.7465), tensor(0.7887), tensor(0.7847), tensor(0.7968), tensor(0.7646), tensor(0.7626), tensor(0.7827), tensor(0.8109), tensor(0.7525), tensor(0.7867), tensor(0.7887), tensor(0.7746), tensor(0.7827), tensor(0.7666), tensor(0.7968), tensor(0.7988), tensor(0.8129), tensor(0.7988), tensor(0.7867), tensor(0.7928), tensor(0.7726), tensor(0.8008), tensor(0.8089), tensor(0.7807), tensor(0.7948), tensor(0.8109), tensor(0.8129), tensor(0.8048), tensor(0.8028), tensor(0.7867), tensor(0.8229), tensor(0.7606), tensor(0.8109), tensor(0.7847), tensor(0.7928), tensor(0.8149), tensor(0.7807), tensor(0.8129), tensor(0.8249), tensor(0.8209), tensor(0.8209), tensor(0.8169), tensor(0.8068), tensor(0.8068), tensor(0.8008), tensor(0.8249), tensor(0.8209), tensor(0.8109), tensor(0.8169), tensor(0.8310), tensor(0.8310), tensor(0.8089), tensor(0.7988), tensor(0.8209), tensor(0.7968), tensor(0.8149), tensor(0.8189), tensor(0.8089), tensor(0.8028), tensor(0.8189), tensor(0.8209), tensor(0.8169), tensor(0.8270), tensor(0.8109), tensor(0.8028), tensor(0.8189), tensor(0.8249), tensor(0.8028), tensor(0.8189)]]

"""

dev_scores = res.split('\n')[1]

test_scores = res.split('\n')[3]

def extract(scores):
    scores = scores[2:-2]
    # dev_scores[-1] = dev_scores[:-2]

    scores = scores.split(', ')


    # print(dev_scores[0])
    # raise
    fold_scores = []
    fold = []
    for score in scores:
        
        
        s = score.split('(')[1]
        s = s[:-1]
        if not s.endswith(')'):
            fold.append(s)
        else:
            fold.append(s[:-1])
            fold_scores.append(fold)
            fold = []

    fold_scores.append(fold)

    int_lists = [[float(x) for x in inner_list] for inner_list in fold_scores]

    tensor = torch.tensor(int_lists)

    return tensor


dev_folds = extract(dev_scores)
test_folds = extract(test_scores)


max_index = dev_folds.max(1)[1]
test_accs = []
for fold, i in zip(test_folds, max_index):

    test_accs.append(fold[i])

print(torch.stack(test_accs).mean(), torch.stack(test_accs).std())