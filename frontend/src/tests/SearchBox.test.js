
import '@testing-library/jest-dom'
import {render, screen , fireEvent} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import {BrowserRouter as Router} from 'react-router-dom';
import SearchBox from '../components/SearchBox';


describe("Search Box", () => {
    test('Check if search box render', () => {

        render(
            <Router>
                <SearchBox />,
            </Router>,
        );

       const searchbox = screen.getByTestId('searchbox-test');
       expect(searchbox).toBeInTheDocument();
    })

    test('Check if search box value changes', () => {

        render(
            <Router>
                <SearchBox />,
            </Router>,
        );

       const searchfield = screen.getByTestId("search-test");
       userEvent.type(searchfield, "test");
       expect(searchfield.value).toMatch("test");
    })

    test('Check if search button is clickable', () => {

        render(
            <Router>
                <SearchBox />,
            </Router>,
        );

       const searchbutton = screen.getByTestId("bttn-test");
       expect(fireEvent.click(searchbutton));
    })
})