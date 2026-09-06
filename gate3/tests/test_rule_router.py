import unittest

from gate3.rule_router import route


class RuleRouterTest(unittest.TestCase):
    def assertCall(self, text: str, reason: str) -> None:
        decision = route(text)
        self.assertTrue(decision.call_refiner, decision)
        self.assertTrue(any(value.startswith(reason) for value in decision.reasons), decision)

    def test_multiple_double_leads_is_selected(self) -> None:
        self.assertCall("装这个监控，其其实也也是为了偷窥。", "multiple_double_leads")
        self.assertCall("装这个监控，其其实也是也是为了偷窥。", "double_lead_with_repeated_bigram")

    def test_explicit_self_correction_is_selected(self) -> None:
        self.assertCall("我不想杀他，的不对，我想杀死他。", "explicit_self_correction")
        self.assertCall("不是红匪，是红军。", "not_but_is_correction")
        self.assertCall("都不能说是水光感，应该是油光感。", "should_be_correction")

    def test_disfluency_triple_is_selected(self) -> None:
        self.assertCall("学习不不不注入血脉。", "triple_disfluency")
        self.assertCall("死死死死你个头啊。", "triple_disfluency")

    def test_common_reduplication_is_not_selected(self) -> None:
        for text in (
            "拜拜。", "好好好吃就吃。", "慢慢的站了起来。", "一个个打开。",
            "战战兢兢地看着他。", "爸爸妈妈都来了。", "酸酸甜甜的米酒。",
            "一个一个一个岗位。",
        ):
            self.assertFalse(route(text).call_refiner, text)

    def test_judgement_or_hedge_alone_is_not_selected(self) -> None:
        for text in ("这一舒服就不对了。", "我不对，刚才对不起了。", "应该是东北的广大农村。"):
            self.assertFalse(route(text).call_refiner, text)

    def test_digits_or_latin_alone_are_not_selected(self) -> None:
        for text in ("IP68 防水防尘。", "一千五百九十九元。", "BHA 是水杨酸。"):
            self.assertFalse(route(text).call_refiner, text)


if __name__ == "__main__":
    unittest.main()
