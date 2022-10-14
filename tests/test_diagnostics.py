import torch

from diagnostics import ImageDiagnostic


class TestDiagnostics:
    def test_image_diagnostic(self):
        bins = torch.linspace(-5, 5, 10)
        screen = ImageDiagnostic(bins)

        x = torch.randn([100]).unsqueeze(0)
        y = torch.randn([100]).unsqueeze(0)

        images = screen.calculate_images(x, y)
        assert images.shape == torch.Size([1, 10, 10])

        x = torch.randn([1000]).reshape(10, 100)
        y = torch.randn([1000]).reshape(10, 100)
        images = screen.calculate_images(x, y)
        assert images.shape == torch.Size([10, 10, 10])
