
import '@testing-library/jest-dom'
import {render, screen , fireEvent} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import {BrowserRouter as Router} from 'react-router-dom';
import CheckoutSteps from '../components/CheckoutSteps';


describe("Checkout Bar Component", () => {

    test('Check if component render', () => {
        render(<CheckoutSteps />);

        const checkmsg = screen.getByTestId('msg-test');
        expect(checkmsg).toBeInTheDocument();
    })

    test('Check if signin text is displayed', () => {
        render(<CheckoutSteps step1/>);

        const checkmsg = screen.getByTestId('signin');
        expect(checkmsg).toBeInTheDocument();
    })

    test('Check if other text is also displayed', () => {
        render(<CheckoutSteps step1/>);
        const checksignin = screen.getByText('Shipping');
        const checkpay = screen.getByText('Payment');
        const checkodr = screen.getByText('Place Order');
        
        expect(checksignin).toBeVisible();
        expect(checkpay).toBeVisible();
        expect(checkodr).toBeVisible();
    })

})