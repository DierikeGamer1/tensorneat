class AmbientDeTreino:
    global contLong, contShort

    def __init__(self, ClosePrice, printar=False):
        self.position_short = False
        self.position_long = False
        self.Final = False
        self.reward = 0
        self.ClosePrice = ClosePrice
        self.printar = printar

    # Método para resetar o estado do ambiente
    def reset(self):
        self.position_short = False
        self.position_long = False
        self.Final = False
        self.reward = 0

        input_list = [
            float(self.position_short),
            float(self.position_long),
            *self.ClosePrice[:200],
        ]
        return input_list

    def actions(self, action, ClosePrice, indice):
        global contShort, contLong
        reward = 0
        # Abrir Long

        if action == 0:
            if self.printar:
                print("Abrir Long")

            reward, self.Final = self.OpenPosition(
                "Long", BuyPrice=ClosePrice[indice])
        # Sleep
        elif action == 1:
            # print("Sleep")
            # if self.position_long and self.PriceOpenLong < ClosePrice[indice]:
            #     reward += ClosePrice[indice] - self.PriceOpenLong
            # elif self.position_long and self.PriceOpenLong > ClosePrice[indice]:
            #     reward -= self.PriceOpenLong - ClosePrice[indice]

            # elif self.position_short and self.PriceOpenShort < ClosePrice[indice]:
            #     reward -= ClosePrice[indice] - self.PriceOpenShort
            # elif self.position_short and self.PriceOpenShort > ClosePrice[indice]:
            #     reward += self.PriceOpenShort - ClosePrice[indice]
            pass

        # Abri Short
        elif action == 2:
            if self.printar:
                print("Abrir Short")

            reward, self.Final = self.OpenPosition(
                "Short", BuyPrice=ClosePrice[indice])
        # Close Long
        elif action == 3:
            if self.printar:
                print("Close Long")
            reward, self.Final = self.ClosePosition(
                "Long", SellPrice=ClosePrice, indice=indice)
            # if reward != 0:
            #     print(reward)
        # Close Short
        elif action == 4:
            if self.printar:
                print("Close Short")
            reward, self.Final = self.ClosePosition(
                "Short", SellPrice=ClosePrice, indice=indice)
            # if reward != 0:
            #     print(reward)
        else:
            if self.printar:
                print("Erro")
            self.Final = True
        return (
            reward,
            self.Final,
            [float(self.position_long), float(self.position_short)],
        )

    def OpenPosition(self, direction, BuyPrice):
        reward = 0
        if direction == "Long":
            if self.position_long or self.position_short:
                self.Final = True
            else:
                self.position_long = True
                self.PriceOpenLong = BuyPrice
                # print(self.PriceOpenLong)
                reward -= 0.0005

        elif direction == "Short":
            if self.position_long or self.position_short:
                self.Final = True
            else:
                self.position_short = True
                self.PriceOpenShort = BuyPrice
                reward -= 0.0005

        return reward, self.Final

    def ClosePosition(self, DirectionPosition, SellPrice, indice):
        reward = 0
        if DirectionPosition == "Long":
            if self.position_long == False:
                self.Final = True
            elif SellPrice[indice] > SellPrice[indice-1]:
                self.Final = True
            else:
                if self.PriceOpenLong < SellPrice[indice]:
                    reward += SellPrice[indice] - self.PriceOpenLong
                else:
                    reward -= self.PriceOpenLong - SellPrice[indice]
            self.position_long = False

        elif DirectionPosition == "Short":
            if self.position_short == False:
                self.Final = True
            elif SellPrice[indice] < SellPrice[indice-1]:
                self.Final = True
            else:
                if self.PriceOpenShort > SellPrice[indice]:
                    reward += self.PriceOpenShort - SellPrice[indice]
                else:
                    reward -= SellPrice[indice] - self.PriceOpenShort
            self.position_short = False
        return reward, self.Final

    def FecharAcabouDados(self, SellPrice, indice):
        reward = 0

        if self.position_long:
            if self.PriceOpenLong < SellPrice[indice]:
                reward += SellPrice[indice] - self.PriceOpenLong
            else:
                reward -= self.PriceOpenLong - SellPrice[indice]

        elif self.position_short:
            if self.PriceOpenShort > SellPrice[indice]:
                reward += self.PriceOpenShort - SellPrice[indice]
            else:
                reward -= SellPrice[indice] - self.PriceOpenShort

        return reward
